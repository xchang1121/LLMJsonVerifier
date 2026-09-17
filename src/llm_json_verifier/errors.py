"""Errors safe to expose without including document text or upstream response bodies."""


class VerifierError(Exception):
    status_code = 422
    code = "invalid_request"


class TokenizerContractError(VerifierError):
    code = "tokenizer_contract"


class CapacityError(VerifierError):
    code = "capacity_exceeded"


class SchemaError(VerifierError):
    code = "invalid_schema"


class BackendError(VerifierError):
    status_code = 502
    code = "backend_error"


class BackendTimeout(BackendError):
    status_code = 504
    code = "backend_timeout"


class BackendBusy(BackendError):
    status_code = 503
    code = "backend_busy"


class BackendProtocolError(BackendError):
    code = "backend_protocol_error"
