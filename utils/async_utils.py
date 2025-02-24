import asyncio
from functools import wraps
import streamlit as st

def ensure_async_loop(func):
    """Decorator to ensure async loop exists"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not hasattr(asyncio, 'get_running_loop'):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return await func(*args, **kwargs)
    return wrapper

def run_async(coro):
    """Run async function in Streamlit context"""
    try:
        loop = st.session_state.async_loop
        return loop.run_until_complete(coro)
    except Exception as e:
        st.error(f"Async operation failed: {str(e)}")
        raise