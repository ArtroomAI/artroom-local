def toast_status(
    title: str, status: str, description: str = "",
    position: str = 'top', duration: int = 5000, isClosable: bool = True, id: str = None):
    return {
        'id': id,
        'title': title,
        'description': description,
        'status': status,
        'position': position,
        'duration': duration,
        'isClosable': isClosable
    }
