#!/usr/bin/env python3
from api_key_crypto import get_api_key




def main() -> None:
    api_key = get_api_key() # handles create-or-load flow and error handling
    # Use the key without printing it
    # Example: os.environ["OPENAI_API_KEY"] = api_key
    print("API key loaded into variable `api_key`.")




if __name__ == "__main__":
    main()
