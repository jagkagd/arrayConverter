class AstSyntaxError(SyntaxError):
    def __init__(self, s: str):
        super().__init__(s)


class LinearError(BaseException):
    def __init__(self):
        super().__init__()


class TransformError(BaseException):
    def __init__(self, s):
        super().__init__(s)
