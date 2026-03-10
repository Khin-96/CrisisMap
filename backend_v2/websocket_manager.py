import asyncio
import json
import logging
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections.add(websocket)
            self.connection_info[websocket] = {
                "connected_at": asyncio.get_event_loop().time(),
                "client_ip": websocket.client.host if websocket.client else "unknown"
            }
            
            logger.info(f"WebSocket connected: {websocket.client}")
            
            # Send welcome message
            await self.send_personal_message({
                "type": "connection_established",
                "message": "Connected to CrisisMap real-time updates",
                "timestamp": asyncio.get_event_loop().time()
            }, websocket)
            
            # Keep connection alive and handle messages
            await self._handle_connection(websocket)
            
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await self.disconnect(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        if websocket in self.connection_info:
            del self.connection_info[websocket]
            
        logger.info(f"WebSocket disconnected: {websocket.client}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def _handle_connection(self, websocket: WebSocket):
        """Handle incoming messages from WebSocket connection"""
        try:
            while True:
                # Wait for message from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await self._process_client_message(message, websocket)
                except json.JSONDecodeError:
                    await self.send_personal_message({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }, websocket)
                
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
            await self.disconnect(websocket)
    
    async def _process_client_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Process messages received from client"""
        message_type = message.get("type")
        
        if message_type == "ping":
            await self.send_personal_message({
                "type": "pong",
                "timestamp": asyncio.get_event_loop().time()
            }, websocket)
            
        elif message_type == "subscribe":
            # Handle subscription to specific data streams
            subscription = message.get("subscription", {})
            await self._handle_subscription(subscription, websocket)
            
        elif message_type == "unsubscribe":
            # Handle unsubscription
            subscription = message.get("subscription", {})
            await self._handle_unsubscription(subscription, websocket)
            
        else:
            await self.send_personal_message({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }, websocket)
    
    async def _handle_subscription(self, subscription: Dict[str, Any], websocket: WebSocket):
        """Handle client subscription to data streams"""
        subscription_type = subscription.get("type")
        
        if subscription_type == "events":
            # Subscribe to real-time event updates
            if websocket not in self.connection_info:
                self.connection_info[websocket] = {}
            
            self.connection_info[websocket]["subscriptions"] = \
                self.connection_info[websocket].get("subscriptions", [])
            
            if "events" not in self.connection_info[websocket]["subscriptions"]:
                self.connection_info[websocket]["subscriptions"].append("events")
            
            await self.send_personal_message({
                "type": "subscription_confirmed",
                "subscription": "events",
                "message": "Subscribed to real-time event updates"
            }, websocket)
            
        elif subscription_type == "alerts":
            # Subscribe to alert notifications
            if websocket not in self.connection_info:
                self.connection_info[websocket] = {}
            
            self.connection_info[websocket]["subscriptions"] = \
                self.connection_info[websocket].get("subscriptions", [])
            
            if "alerts" not in self.connection_info[websocket]["subscriptions"]:
                self.connection_info[websocket]["subscriptions"].append("alerts")
            
            await self.send_personal_message({
                "type": "subscription_confirmed",
                "subscription": "alerts",
                "message": "Subscribed to alert notifications"
            }, websocket)
    
    async def _handle_unsubscription(self, subscription: Dict[str, Any], websocket: WebSocket):
        """Handle client unsubscription from data streams"""
        subscription_type = subscription.get("type")
        
        if websocket in self.connection_info:
            subscriptions = self.connection_info[websocket].get("subscriptions", [])
            
            if subscription_type in subscriptions:
                subscriptions.remove(subscription_type)
                
                await self.send_personal_message({
                    "type": "unsubscription_confirmed",
                    "subscription": subscription_type,
                    "message": f"Unsubscribed from {subscription_type}"
                }, websocket)
    
    async def broadcast_event_update(self, event_data: Dict[str, Any]):
        """Broadcast new event to subscribed clients"""
        message = {
            "type": "event_update",
            "data": event_data,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Send to clients subscribed to events
        subscribed_connections = [
            conn for conn in self.active_connections
            if self.connection_info.get(conn, {}).get("subscriptions", []) and
            "events" in self.connection_info[conn]["subscriptions"]
        ]
        
        for connection in subscribed_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send event update: {e}")
                await self.disconnect(connection)
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert to subscribed clients"""
        message = {
            "type": "alert",
            "data": alert_data,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Send to clients subscribed to alerts
        subscribed_connections = [
            conn for conn in self.active_connections
            if self.connection_info.get(conn, {}).get("subscriptions", []) and
            "alerts" in self.connection_info[conn]["subscriptions"]
        ]
        
        for connection in subscribed_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
                await self.disconnect(connection)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_subscriptions": sum(
                len(info.get("subscriptions", []))
                for info in self.connection_info.values()
            ),
            "connections_by_subscription": {
                "events": len([
                    conn for conn in self.active_connections
                    if "events" in self.connection_info.get(conn, {}).get("subscriptions", [])
                ]),
                "alerts": len([
                    conn for conn in self.active_connections
                    if "alerts" in self.connection_info.get(conn, {}).get("subscriptions", [])
                ])
            }
        }