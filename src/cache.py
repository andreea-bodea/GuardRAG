from joblib.memory import Memory

cache = Memory(location=".cache", verbose=0).cache
