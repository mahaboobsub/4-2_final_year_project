try:
    import langchain_classic
    print("langchain_classic exists")
except ImportError:
    print("langchain_classic does NOT exist")

try:
    from langchain.chains import ConversationalRetrievalChain
    print("langchain.chains.ConversationalRetrievalChain exists")
except ImportError:
    print("langchain.chains.ConversationalRetrievalChain does NOT exist")
