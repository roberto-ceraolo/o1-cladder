
import pdb
from typing import Any, Dict, List, Optional
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import backoff
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tqdm import tqdm

import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


# Configure logging
from datetime import datetime

log_filename = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

pricing = json.load(open("api_pricing.json"))


class TokenUsage(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    completion_tokens_details: Optional[Dict[str, int]] = None
    cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert TokenUsage to a dictionary for JSON serialization"""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_tokens_details": self.completion_tokens_details,
            "cost": self.cost
        }


class ResponseFormat(BaseModel):
    answer: str = Field(description="The parsed boolean answer")
    reasoning: str = Field(description="The reasoning for the answer")

class ProcessingConfig:
    def __init__(
        self,
        inference_model: str = "",
        parsing_model: str = "gpt-4o-mini",
        max_workers: int = 5,
        max_retries: int = 3,
        initial_wait: float = 1,
        max_wait: float = 60
    ):
        self.inference_model = inference_model
        self.parsing_model = parsing_model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.max_wait = max_wait

class DataProcessor:
    def __init__(self, client: OpenAI, config: ProcessingConfig):
        self.client = client
        self.config = config
        
    def construct_prompt(self, row: Dict[str, Any]) -> str:
        """Constructs the prompt from the data fields"""
        return f"""{row.get('background', '')}
        \n\n
{row.get('given_info', '')}
\n\n
{row.get('question', '')}"""

    @backoff.on_exception(
        backoff.expo,
        RateLimitError,
        max_tries=3,
        max_time=300
    )
    def get_inference(self, prompt: str) -> Dict[str, Any]:
        """Makes the API call to the inference model with backoff"""
        system_prompt = '''
You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content. Reason about it and then give a "yes" or "no" final answer. Before giving the answer, show your reasoning.
        '''.strip()

        try:
            response = self.client.chat.completions.create(
                model=self.config.inference_model,
                messages=[
                    {"role": "user", "content": system_prompt + "\n\n" + prompt}
                ],
                max_completion_tokens=5120
            )
            
            model_pricing = pricing[self.config.inference_model]
            cost = response.usage.prompt_tokens * model_pricing["input_tokens_per_1M"] / 1000000 + response.usage.completion_tokens * model_pricing["output_tokens_per_1M"] / 1000000
            usage = TokenUsage(
                total_tokens=response.usage.total_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cost=cost
            )
            # log the cost
            logger.info(f"Inference cost: {cost}")
            
            return {
                "content": response.choices[0].message.content,
                "usage": usage
            }
        except Exception as e:
            logger.error(f"Error in inference: {str(e)}")
            raise

    @backoff.on_exception(
        backoff.expo,
        RateLimitError,
        max_tries=3,
        max_time=300
    )
    def parse_response(self, content: str, response_format: BaseModel) -> Dict[str, Any]:
        """Parses the response using the parsing model"""
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.config.parsing_model,
                messages=[
                    {"role": "user", "content": f"Parse the following response: {content}, returning the boolean answer ('yes' or 'no') and the reasoning for the answer."}
                ],
                response_format=response_format
            )
            
            model_pricing = pricing[self.config.parsing_model]
            return {
                "parsed": response.choices[0].message.parsed,
                "usage": TokenUsage(
                    total_tokens=response.usage.total_tokens,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost=response.usage.prompt_tokens * model_pricing["input_tokens_per_1M"] / 1000000 + response.usage.completion_tokens * model_pricing["output_tokens_per_1M"] / 1000000
                )
            }
        except Exception as e:
            logger.error(f"Error in parsing: {str(e)}")
            raise

    def process_single_item(self, row: Dict[str, Any], response_format: BaseModel) -> Dict[str, Any]:
        """Processes a single item through both inference and parsing"""
        prompt = self.construct_prompt(row)
        
        inference_result = self.get_inference(prompt)
        parsed_result = self.parse_response(inference_result["content"], response_format)
        
        return {
            "prompt": prompt,
            "question_id": row["question_id"],
            "raw_inference_result": inference_result["content"],
            "parsed_result_answer": parsed_result["parsed"].answer,
            "parsed_result_reasoning": parsed_result["parsed"].reasoning,
            "usage": {
                "inference": inference_result["usage"],
                "parsing": parsed_result["usage"]
            }
        }

class ResultWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def write_result(self, result: Dict[str, Any]):
        """Writes a single result to the output file"""
        # Convert TokenUsage objects to dictionaries before JSON serialization
        if "usage" in result:
            result["usage"] = {
                "inference": result["usage"]["inference"].to_dict(),
                "parsing": result["usage"]["parsing"].to_dict()
            }
        with self.output_path.open('a') as f:
            json.dump(result, f)
            f.write('\n')

def load_retry_ids(retry_file_path: str) -> List[str]:
    """Load question IDs to retry from a file"""
    with open(retry_file_path) as f:
        return [int(line.strip()) for line in f]

def get_last_processed_id(output_path: Path) -> Optional[str]:
    """Returns the question_id of the last processed item if the file exists"""
    if not output_path.exists():
        return None
    
    try:
        with output_path.open('r') as f:
            # Move to the end of file
            f.seek(0, 2)
            pos = f.tell()
            
            # Read backwards until we find the last valid line
            while pos > 0:
                pos -= 1
                f.seek(pos)
                if f.read(1) == '\n':
                    last_line = f.readline()
                    if pos != 0:
                        try:
                            result = json.loads(last_line)
                            return result.get('question_id')
                        except json.JSONDecodeError:
                            continue
            
            # Handle case where file has only one line
            f.seek(0)
            try:
                result = json.loads(f.readline())
                return result.get('question_id')
            except json.JSONDecodeError:
                return None
    except Exception as e:
        logger.error(f"Error reading last processed id: {str(e)}")
        return None

async def process_dataset(
    data_path: str,
    output_path: str,
    response_format: BaseModel,
    config: Optional[ProcessingConfig] = None,
    retry_ids: Optional[List[str]] = None 
):
    """Main function to process the dataset"""
    if config is None:
        config = ProcessingConfig()
        
    client = OpenAI()
    processor = DataProcessor(client, config)
    output_path = Path(output_path)
    writer = ResultWriter(output_path)
    
    # Load dataset
    dataset = []
    with open(data_path) as f:
        for line in f:
            dataset.append(json.loads(line.strip()))

    # Filter dataset based on retry_ids if provided
    if retry_ids:
        pdb.set_trace()
        dataset = [row for row in dataset if row['question_id'] in retry_ids]
        logger.info(f"Processing {len(dataset)} questions from retry list")
    else:
        # Existing resume logic
        last_processed_id = get_last_processed_id(output_path)
        if last_processed_id:
            dataset = [row for row in dataset if row['question_id'] > last_processed_id]
            logger.info(f"Resuming from question_id: {last_processed_id}")

    # print the number of rows
    print(f"Processing {len(dataset)} rows")
    
    # Process items using ThreadPoolExecutor instead of asyncio.gather
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(processor.process_single_item, row, response_format): row 
            for row in dataset
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
            try:
                result = future.result()
                writer.write_result(result)
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")


if __name__ == "__main__":
    import asyncio
    
    inference_model = "o1-mini-2024-09-12"
    parsing_model = "gpt-4o-mini-2024-07-18"

    config = ProcessingConfig(
        inference_model=inference_model,
        parsing_model=parsing_model,
        max_workers=4
    )

    retry_ids = load_retry_ids("missing_ids_o1-mini.txt")
    root_path = "../../2304_caubench/data/cladder-v1/"
    dataset_name = "o1-preview-data-cladder-v1-q-balanced_rand-top1000.json"
    asyncio.run(process_dataset(
        data_path=root_path + dataset_name,
        output_path=f"results-{inference_model}-{dataset_name.split('.')[0]}.jsonl",
        response_format=ResponseFormat,
        config=config,
        retry_ids=retry_ids
    ))
