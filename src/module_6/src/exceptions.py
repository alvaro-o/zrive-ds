from fastapi import HTTPException
from fastapi import status

class UserNotFoundException(HTTPException):
    def __init__(self, detail: str = "User not found"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )


class PredictionException(HTTPException):
    def __init__(self, detail: str = "Prediction failed"):
        super().__init__(
            status_code=500,
            detail=detail,
        )