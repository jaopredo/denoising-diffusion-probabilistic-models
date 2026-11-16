from typing import Any, TypeVar

T = TypeVar("T")

class SingletonMeta(type):
    _instances: dict[type, Any] = {}

    def __call__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in SingletonMeta._instances:
            SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)

        return SingletonMeta._instances[cls]