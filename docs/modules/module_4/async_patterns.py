#!/usr/bin/env python3
"""
MCP Asynchronous Processing Patterns
===================================

This module demonstrates various patterns for handling asynchronous
context execution in MCP client applications, including callbacks,
promises, polling strategies, and WebSocket integration.

Key components:
- Callback-based execution
- Promise pattern for async processing
- Polling strategies with backoff
- WebSocket-based real-time updates
"""

import time
import json
import uuid
import asyncio
import threading
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

try:
    import websocket
except ImportError:
    print("WebSocket support requires 'websocket-client' package. Install with: pip install websocket-client")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_async")

# Type definitions
T = TypeVar('T')
ContextResult = Dict[str, Any]
ContextCallback = Callable[[str, ContextResult], None]


# ===== Callback Pattern =====

class CallbackBasedClient:
    """
    Demonstrates callback-based async processing.
    
    This pattern registers callbacks to be executed when a context
    completes, allowing for non-blocking operation.
    """
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.pending_contexts = {}  # context_id -> callback
        
        # In a real implementation, this would handle HTTP requests
        # For demo purposes, we'll simulate async processing
        self.worker_thread = threading.Thread(target=self._process_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def execute_context(self, 
                       context_name: str, 
                       params: Dict[str, Any], 
                       callback: ContextCallback) -> str:
        """
        Execute a context with a callback for completion.
        
        Args:
            context_name: Name of the context to execute
            params: Parameters for the context
            callback: Function to call when context completes
            
        Returns:
            context_id: Unique ID for the context execution
        """
        # Generate context ID
        context_id = str(uuid.uuid4())
        
        # Store callback
        self.pending_contexts[context_id] = callback
        
        # In a real implementation, this would make an API request
        # For demo purposes, we'll just log and return
        logger.info(f"Executing context {context_name} with ID {context_id}")
        
        return context_id
    
    def _process_worker(self):
        """Background worker that simulates context completion."""
        while True:
            time.sleep(1)  # Check every second
            
            # Make a copy to avoid modification during iteration
            contexts = list(self.pending_contexts.items())
            
            for context_id, callback in contexts:
                # Simulate random completion (10% chance each second)
                if uuid.uuid4().int % 10 == 0:
                    # Generate mock result
                    result = {
                        "context_id": context_id,
                        "status": "completed",
                        "result": {"message": "Context execution completed successfully"},
                        "execution_time_ms": 1500
                    }
                    
                    # Remove from pending
                    del self.pending_contexts[context_id]
                    
                    # Execute callback
                    try:
                        callback(context_id, result)
                    except Exception as e:
                        logger.error(f"Error in callback for context {context_id}: {str(e)}")


# ===== Promise Pattern =====

class ContextState(Enum):
    """States for a context promise."""
    PENDING = "pending"
    FULFILLED = "fulfilled"
    REJECTED = "rejected"


class ContextPromise(Generic[T]):
    """
    Promise-like pattern for asynchronous context execution.
    
    This implementation provides a Future-like API for working with async results.
    """
    
    def __init__(self, context_id: str):
        self.context_id = context_id
        self.state = ContextState.PENDING
        self.result = None
        self.error = None
        self._then_callbacks = []
        self._catch_callbacks = []
        self._finally_callbacks = []
        self._condition = threading.Condition()
    
    def then(self, callback: Callable[[T], Any]) -> 'ContextPromise':
        """Register a callback for successful completion."""
        with self._condition:
            if self.state == ContextState.FULFILLED:
                try:
                    callback(self.result)
                except Exception as e:
                    logger.error(f"Error in 'then' callback: {str(e)}")
            else:
                self._then_callbacks.append(callback)
        return self
    
    def catch(self, callback: Callable[[Exception], Any]) -> 'ContextPromise':
        """Register a callback for errors."""
        with self._condition:
            if self.state == ContextState.REJECTED:
                try:
                    callback(self.error)
                except Exception as e:
                    logger.error(f"Error in 'catch' callback: {str(e)}")
            else:
                self._catch_callbacks.append(callback)
        return self
    
    def finally_do(self, callback: Callable[[], Any]) -> 'ContextPromise':
        """Register a callback to run after completion (success or error)."""
        with self._condition:
            if self.state != ContextState.PENDING:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in 'finally' callback: {str(e)}")
            else:
                self._finally_callbacks.append(callback)
        return self
    
    def resolve(self, result: T) -> None:
        """Resolve the promise with a successful result."""
        with self._condition:
            if self.state != ContextState.PENDING:
                return
            
            self.state = ContextState.FULFILLED
            self.result = result
            
            # Notify waiters
            self._condition.notify_all()
        
        # Execute callbacks outside the lock
        for callback in self._then_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in 'then' callback: {str(e)}")
        
        for callback in self._finally_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in 'finally' callback: {str(e)}")
    
    def reject(self, error: Exception) -> None:
        """Reject the promise with an error."""
        with self._condition:
            if self.state != ContextState.PENDING:
                return
            
            self.state = ContextState.REJECTED
            self.error = error
            
            # Notify waiters
            self._condition.notify_all()
        
        # Execute callbacks outside the lock
        for callback in self._catch_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in 'catch' callback: {str(e)}")
        
        for callback in self._finally_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in 'finally' callback: {str(e)}")
    
    def wait(self, timeout: Optional[float] = None) -> T:
        """Wait for the promise to be fulfilled and return the result."""
        with self._condition:
            if self.state == ContextState.PENDING:
                self._condition.wait(timeout=timeout)
            
            if self.state == ContextState.FULFILLED:
                return self.result
            elif self.state == ContextState.REJECTED:
                raise self.error
            else:
                raise TimeoutError(f"Timed out waiting for context {self.context_id}")


class PromiseBasedClient:
    """
    Demonstrates promise-based async processing.
    
    This pattern returns a Promise-like object that represents
    the future result of the async operation.
    """
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.promises = {}  # context_id -> ContextPromise
        
        # In a real implementation, this would handle HTTP requests
        # For demo purposes, we'll simulate async processing
        self.worker_thread = threading.Thread(target=self._process_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def execute_context(self, 
                       context_name: str, 
                       params: Dict[str, Any]) -> ContextPromise:
        """
        Execute a context and return a promise.
        
        Args:
            context_name: Name of the context to execute
            params: Parameters for the context
            
        Returns:
            promise: Promise that will resolve with the context result
        """
        # Generate context ID
        context_id = str(uuid.uuid4())
        
        # Create promise
        promise = ContextPromise(context_id)
        self.promises[context_id] = promise
        
        # In a real implementation, this would make an API request
        # For demo purposes, we'll just log and return
        logger.info(f"Executing context {context_name} with ID {context_id}")
        
        return promise
    
    def _process_worker(self):
        """Background worker that simulates context completion."""
        while True:
            time.sleep(1)  # Check every second
            
            # Make a copy to avoid modification during iteration
            promises = list(self.promises.items())
            
            for context_id, promise in promises:
                # Simulate random completion (10% chance each second)
                if uuid.uuid4().int % 10 == 0:
                    # Generate mock result
                    result = {
                        "context_id": context_id,
                        "status": "completed",
                        "result": {"message": "Context execution completed successfully"},
                        "execution_time_ms": 1500
                    }
                    
                    # Remove from pending
                    del self.promises[context_id]
                    
                    # Resolve promise
                    promise.resolve(result)
                
                # Simulate random error (5% chance each second)
                elif uuid.uuid4().int % 20 == 0:
                    # Remove from pending
                    del self.promises[context_id]
                    
                    # Reject promise
                    promise.reject(Exception("Context execution failed"))


# ===== Polling Pattern =====

class PollingClient:
    """
    Demonstrates polling-based async processing.
    
    This pattern polls for updates at a variable rate with
    exponential backoff to reduce server load.
    """
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.context_states = {}  # context_id -> status
        
        # In a real implementation, this would handle actual HTTP requests
    
    def execute_context(self, 
                       context_name: str, 
                       params: Dict[str, Any]) -> str:
        """
        Execute a context and return its ID for polling.
        
        Args:
            context_name: Name of the context to execute
            params: Parameters for the context
            
        Returns:
            context_id: Unique ID for the context execution
        """
        # Generate context ID
        context_id = str(uuid.uuid4())
        
        # Store initial state
        self.context_states[context_id] = "created"
        
        # In a real implementation, this would make an API request
        # For demo purposes, we'll just log and return
        logger.info(f"Executing context {context_name} with ID {context_id}")
        
        return context_id
    
    def poll_until_complete(self, 
                           context_id: str,
                           initial_interval: float = 0.5,
                           max_interval: float = 10.0,
                           timeout: float = 60.0,
                           backoff_factor: float = 1.5) -> Dict[str, Any]:
        """
        Poll for context completion with exponential backoff.
        
        Args:
            context_id: The context ID to poll
            initial_interval: Initial polling interval in seconds
            max_interval: Maximum polling interval in seconds
            timeout: Maximum time to wait in seconds
            backoff_factor: Factor to increase wait time by after each poll
            
        Returns:
            result: The final context result
            
        Raises:
            TimeoutError: If the operation times out
        """
        if context_id not in self.context_states:
            raise ValueError(f"Unknown context ID: {context_id}")
        
        interval = initial_interval
        start_time = time.time()
        elapsed = 0
        
        # In a real implementation, this would make actual HTTP requests
        # For demo, we'll simulate with random state changes
        
        while elapsed < timeout:
            # Get current state
            current_state = self.context_states.get(context_id)
            
            # Check if complete
            if current_state in ["completed", "failed", "canceled"]:
                # Return final result
                return {
                    "context_id": context_id,
                    "status": current_state,
                    "result": {"message": "Context execution completed"} if current_state == "completed" else None,
                    "error": "Context execution failed" if current_state == "failed" else None,
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }
            
            # Wait for next interval
            time.sleep(interval)
            elapsed = time.time() - start_time
            
            # Simulate state change
            self._simulate_state_change(context_id)
            
            # Increase interval with backoff (up to max_interval)
            interval = min(interval * backoff_factor, max_interval)
        
        # Timeout reached
        raise TimeoutError(f"Timed out polling for context {context_id}")
    
    def _simulate_state_change(self, context_id: str):
        """Simulate a state change for demo purposes."""
        current_state = self.context_states.get(context_id)
        
        if current_state == "created":
            self.context_states[context_id] = "executing"
        elif current_state == "executing":
            # 30% chance to complete
            if uuid.uuid4().int % 10 < 3:
                self.context_states[context_id] = "completed"
            # 10% chance to fail
            elif uuid.uuid4().int % 10 == 9:
                self.context_states[context_id] = "failed"


# ===== WebSocket Pattern =====

class WebSocketClient:
    """
    Demonstrates WebSocket-based async processing.
    
    This pattern uses WebSockets for real-time updates on
    context execution status.
    """
    
    def __init__(self, server_url: str, websocket_url: str = None):
        self.server_url = server_url
        self.websocket_url = websocket_url or server_url.replace("http", "ws") + "/ws"
        self.ws = None
        self.connected = False
        self.listeners = {}  # context_id -> list of callbacks
        self.global_listeners = []  # callbacks for all events
        
        # Connect WebSocket in background
        self.connect_thread = threading.Thread(target=self._connect_websocket)
        self.connect_thread.daemon = True
        self.connect_thread.start()
    
    def _connect_websocket(self):
        """Connect to WebSocket server."""
        try:
            # In a real implementation, this would connect to a real server
            # For demo purposes, we'll simulate with a mock WebSocket
            
            # Create mock handlers
            def on_message(ws, message):
                self._handle_message(message)
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {str(error)}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"WebSocket closed: {close_msg}")
                self.connected = False
                
                # Try to reconnect after delay
                time.sleep(5)
                self._connect_websocket()
            
            def on_open(ws):
                logger.info("WebSocket connected")
                self.connected = True
            
            # Create and connect WebSocket
            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Run WebSocket connection (blocks this thread)
            self.ws.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            self.connected = False
            
            # Try to reconnect after delay
            time.sleep(5)
            self._connect_websocket()
    
    def _handle_message(self, message):
        """Handle incoming WebSocket message."""
        try:
            # Parse message
            data = json.loads(message)
            
            # Get context ID
            context_id = data.get("context_id")
            
            # Notify global listeners
            for listener in self.global_listeners:
                try:
                    listener(data)
                except Exception as e:
                    logger.error(f"Error in global listener: {str(e)}")
            
            # Notify context-specific listeners
            if context_id and context_id in self.listeners:
                for listener in self.listeners[context_id]:
                    try:
                        listener(data)
                    except Exception as e:
                        logger.error(f"Error in context listener: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
    
    def execute_context(self, 
                       context_name: str, 
                       params: Dict[str, Any]) -> str:
        """
        Execute a context and return its ID.
        
        In a real implementation, this would make an HTTP request
        to start the context execution, then listen for updates
        over WebSocket.
        """
        # Generate context ID
        context_id = str(uuid.uuid4())
        
        # In a real implementation, this would make an API request
        # For demo purposes, we'll just log and simulate
        logger.info(f"Executing context {context_name} with ID {context_id}")
        
        # Simulate sending message to server
        if self.connected and self.ws:
            self.ws.send(json.dumps({
                "action": "execute_context",
                "context_name": context_name,
                "params": params
            }))
        
        # Start simulation for demo purposes
        self._simulate_context_updates(context_id)
        
        return context_id
    
    def add_context_listener(self, context_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a listener for updates on a specific context."""
        if context_id not in self.listeners:
            self.listeners[context_id] = []
        self.listeners[context_id].append(callback)
    
    def add_global_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a listener for all context updates."""
        self.global_listeners.append(callback)
    
    def _simulate_context_updates(self, context_id: str):
        """Simulate context updates for demo purposes."""
        def send_update(state, data=None, error=None):
            message = {
                "context_id": context_id,
                "status": state,
                "timestamp": datetime.now().isoformat()
            }
            
            if data:
                message["result"] = data
            
            if error:
                message["error"] = error
            
            # Simulate receiving message from server
            self._handle_message(json.dumps(message))
        
        # Run in a separate thread
        def update_thread():
            # Created
            send_update("created")
            time.sleep(1)
            
            # Executing
            send_update("executing")
            time.sleep(2)
            
            # 80% chance of success, 20% chance of failure
            if uuid.uuid4().int % 5 < 4:
                # Completed
                send_update("completed", {"message": "Context execution completed successfully"})
            else:
                # Failed
                send_update("failed", None, "Context execution failed")
        
        # Start thread
        thread = threading.Thread(target=update_thread)
        thread.daemon = True
        thread.start()


# ===== Usage Examples =====

def callback_example():
    """Example of callback-based async processing."""
    client = CallbackBasedClient("https://example.com/mcp")
    
    def on_complete(context_id, result):
        print(f"Context {context_id} completed:")
        print(json.dumps(result, indent=2))
    
    context_id = client.execute_context(
        "quickbooks.getInvoices",
        {"customer_id": "12345"},
        on_complete
    )
    
    print(f"Started context execution with ID: {context_id}")
    print("Continuing with other work while context executes...")
    
    # In a real application, your code would continue here
    # The callback will be called when the context completes
    
    # For demo purposes, wait a bit to let the simulated execution complete
    time.sleep(10)


def promise_example():
    """Example of promise-based async processing."""
    client = PromiseBasedClient("https://example.com/mcp")
    
    promise = client.execute_context(
        "quickbooks.getInvoices",
        {"customer_id": "12345"}
    )
    
    # Register callbacks
    promise.then(lambda result: print(f"Success: {json.dumps(result, indent=2)}"))
    promise.catch(lambda error: print(f"Error: {str(error)}"))
    promise.finally_do(lambda: print("Context execution finished"))
    
    print("Continuing with other work while context executes...")
    
    # In a real application, your code would continue here
    # The callbacks will be called when the context completes
    
    # For demo purposes, wait a bit to let the simulated execution complete
    time.sleep(10)


def polling_example():
    """Example of polling-based async processing."""
    client = PollingClient("https://example.com/mcp")
    
    context_id = client.execute_context(
        "quickbooks.getInvoices",
        {"customer_id": "12345"}
    )
    
    print(f"Started context execution with ID: {context_id}")
    
    try:
        # Poll with exponential backoff
        result = client.poll_until_complete(
            context_id,
            initial_interval=0.5,
            max_interval=5.0,
            timeout=30.0
        )
        
        print(f"Context completed: {json.dumps(result, indent=2)}")
        
    except TimeoutError as e:
        print(f"Timeout: {str(e)}")


def websocket_example():
    """Example of WebSocket-based async processing."""
    client = WebSocketClient("https://example.com/mcp")
    
    # Add global listener for all contexts
    client.add_global_listener(lambda data: print(f"Global update: {data.get('status')}"))
    
    # Execute context
    context_id = client.execute_context(
        "quickbooks.getInvoices",
        {"customer_id": "12345"}
    )
    
    # Add listener for this specific context
    client.add_context_listener(
        context_id,
        lambda data: print(f"Context {context_id} update: {data.get('status')}")
    )
    
    print(f"Started context execution with ID: {context_id}")
    print("Continuing with other work while listening for updates...")
    
    # For demo purposes, wait a bit to let the simulated execution complete
    time.sleep(10)


if __name__ == "__main__":
    # This is just a demonstration - not meant to be run directly
    print("This module demonstrates async patterns for MCP clients.")
    print("Import and use the classes in your own code instead of running this file directly.") 