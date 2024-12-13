import os
from dotenv import load_dotenv
load_dotenv()

import requests
from openai import OpenAI
from anthropic import Anthropic
from typing import Dict, Any, Optional

def call_openai(
        prompt: str,
        model: str = "gpt-4o-2024-11-20",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
) -> str:
    """
    Call OpenAI API with the given prompt
    
    Args:
        prompt: Input text prompt
        model: OpenAI model to use
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response (optional)
    
    Returns:
        Generated response text
    """
    try:
        client = OpenAI(api_key=os.getenv("openai_api_key"))
        
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")
    
def call_claude(
        prompt: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
) -> str:
    """
    Call Anthropic Claude API with the given prompt
    
    Args:
        prompt: Input text prompt
        model: Claude model to use
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response (optional, defaults to 1024)
    
    Returns:
        Generated response text
    """
    try:
        client = Anthropic(api_key=os.getenv("anthropic_api_key"))
        
        if max_tokens is None:
            max_tokens = 1024
        
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
            
        response = client.messages.create(**params)
        return response.content[0].text
        
    except Exception as e:
        raise Exception(f"Claude API error: {str(e)}")

def call_openrouter(
        prompt: str,
        model: str = "meta-llama/llama-3.2-3b-instruct",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
) -> str:
    """
    Call OpenRouter API with the given prompt
    
    Args:
        prompt: Input text prompt
        model: Full model path (e.g. "anthropic/claude-3-sonnet")
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response (optional)
    
    Returns:
        Generated response text
    """
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("openrouter_api_key"),
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/your-repo",
                "X-Title": "Algorithm Solver"
            }
        )
        
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"OpenRouter API error: {str(e)}")

def call_perplexity(
        prompt: str,
        model: str = "llama-3.1-sonar-small-128k-online",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        system_prompt: str = "Be precise and concise."
) -> str:
    """
    Call Perplexity API with the given prompt
    
    Args:
        prompt: Input text prompt
        model: Perplexity model to use
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response (optional)
        system_prompt: System prompt to guide the model's behavior
    
    Returns:
        Generated response text
    """
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('perplexity_api_key')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": 0.9,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0.1
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            error_detail = response.json() if response.text else "No error details available"
            raise Exception(f"API returned status code {response.status_code}. Details: {error_detail}")
            
        return response.json()["choices"][0]["message"]["content"]
        
    except Exception as e:
        if isinstance(e, requests.exceptions.RequestException):
            raise Exception(f"Perplexity API network error: {str(e)}")
        raise Exception(f"Perplexity API error: {str(e)}")

def call_llm(
    prompt: str,
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Unified function to call different LLM providers
    
    Args:
        prompt: Input text prompt
        provider: API provider ("openai", "claude", "openrouter", "perplexity")
        model: Model name (provider-specific)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response (optional)
    
    Returns:
        Generated response text
    """

    default_models = {
        "openai": "gpt-4o-2024-11-20",
        "claude": "claude-3-5-sonnet-20241022",
        "openrouter": "meta-llama/llama-3.2-3b-instruct",
        "perplexity": "llama-3.1-sonar-small-128k-online"
    }

    if not model:
        model = default_models.get(provider)
        if not model:
            raise ValueError(f"Unknown provider: {provider}")
    
    if provider == "claude" and max_tokens is None:
        max_tokens = 1024
    
    if provider == "openai":
        return call_openai(prompt, model, temperature, max_tokens)
    elif provider == "claude":
        return call_claude(prompt, model, temperature, max_tokens)
    elif provider == "openrouter":
        return call_openrouter(prompt, model, temperature, max_tokens)
    elif provider == "perplexity":
        return call_perplexity(prompt, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
if __name__ == "__main__":
    prompt = "What is forward guidance in macroeconomics?"

    response_openai = call_llm(prompt, provider="openai", model="gpt-4o-2024-11-20")
    print(f"OpenAI Response: {response_openai}")

    response_claude = call_llm(prompt, provider="claude", model="claude-3-5-sonnet-20241022")
    print(f"\nClaude Response: {response_claude}")

    response_openrouter = call_llm(prompt, provider="openrouter", model="meta-llama/llama-3.2-3b-instruct")
    print(f"\nOpenRouter Response: {response_openrouter}")

    response_perplexity = call_llm(prompt, provider="perplexity")
    print(f"\nPerplexity Response: {response_perplexity}")
