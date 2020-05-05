def get_product_id(mapping):
    def func(x):
        return mapping.get(x, -1)
    return func