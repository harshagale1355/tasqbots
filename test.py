from Chatbot.graph.graph import app


query = "what is the return policy"


result = app.invoke({
    "query": query
})


print("\nANSWER:\n")
print(result["answer"])


print("\nCITATIONS:\n")

for citation in result["citations"]:
    print(citation)