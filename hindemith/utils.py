def symbols(symbol_table):
    def wrapper(fn):
        fn._hm_symbols = symbol_table
        return fn
    return wrapper
