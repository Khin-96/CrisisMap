from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

TOKEN_HEADER = "Authorization"
VALID_TOKENS = {
    # In real deployments, these would come from a secure vault or environment variable
}


class SimpleTokenAuth(HTTPBearer):
    async def __call__(self, request: Request) -> str:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        token = credentials.credentials
        if token not in VALID_TOKENS:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return token


def require_authentication():
    # Placeholder for compatibility with FastAPI dependencies
    return SimpleTokenAuth(auto_error=True)


__all__ = ["require_authentication"]
