import os 
import boto3
import logging 
import OpenAI
import Groq


# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # For CloudWatch visibility


# ensure that handlers are only added from once in aws lambda 
if not logging.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logging.addHandler(handler)


# AWS + openai /groq client 
s3 = boto3.client("s3")
# we have already added these to the cli 
PROVIDER = os.getenv("LLM_PROVIDER" , "groq").lower()
BASE_URL = "https://api.groq.com/openai/v1" if PROVIDER == "groq" else "https://api.openai.com/v1"
API_KEY = os.getenv("GROQ_API_KEY") if PROVIDER == "groq" else os.getenv("OPENAI_API_KEY")
MODEL = "llama3-70b-8192" if PROVIDER == "groq" else "gpt-3.5-turbo"
BUCKET = os.getenv("BUCKET_NAME", "lambda-genai-bucket-3")
logger.info(f"LLM Provider: {PROVIDER}, Model: {MODEL}, Bucket: {BUCKET}")

client = OpenAI(api_key = API_KEY , base_url = BASE_URL)

def summarize_text(text):
    logger.info("Starting text summarization. Text length: %d characters", len(text))
    try:
        response = client.chat_completions.create(
            model = MODEL ,
            messages = [
                {"role": "system", "content": "You are a helpful text summarizer."},
                {"role": "user", "content": f"Summarize this text:\n\n{text}"}
            ],
            temperature = 0.7,
            max_tokens = 2000
        )
        logger.info("Summarization completed.")
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Failed during summarization: %s", str(e), exc_info=True)
        raise


def lambda_handler(event , context):
    logging.info("Lambda triggered. Raw event: %s", event)
    try:
        record = event['Records'][0]
        bucket_name =  record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        logger.info("Received file: s3://%s/%s", bucket_name, key)

        if not key.endswith(".txt"):
            logger.warning("Skipping non-text file: %s", key)
            return {"status": "skipped_non_txt"}
        file_obj = s3.get_object(Bucket=bucket_name, Key=key)
        text = file_obj["Body"].read().decode("utf-8")
        logger.info("Downloaded file. Size: %d characters", len(text))
        
        summary = summarize_text(text)
        
        filename = key.split("/")[-1]
        output_key = f"output/summary_{filename}"
        #W put_object is used to save file in s3
        s3.put_object(Body=summary.encode("utf-8"), Bucket=bucket_name, Key=output_key)

        logger.info("Successfully saved summary to: s3://%s/%s", bucket_name, output_key)
        return {"status": "complete"}
    
    
    except Exception as e:
        logger.error("Unhandled exception: %s", str(e), exc_info=True)
        raise







    except Exception as e:
        logging.error("unexpected exception : %s", str(e) ,exc_info = True)
        raise 


