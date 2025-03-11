import streamlit as st
import time
import hashlib
import functools

def generate_cache_key(*args, **kwargs):
    """Generate a cache key from function arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
    return key

def cache_with_expiry(ttl_seconds=300):
    """Cache function results with expiry time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"cache_{func.__name__}_{generate_cache_key(*args, **kwargs)}"
            
            # Check if result is in cache and not expired
            if cache_key in st.session_state:
                cached_time, result = st.session_state[cache_key]
                if time.time() - cached_time < ttl_seconds:
                    return result
            
            # Calculate result and cache it
            result = func(*args, **kwargs)
            st.session_state[cache_key] = (time.time(), result)
            return result
        return wrapper
    return decorator