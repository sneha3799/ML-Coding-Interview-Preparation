# Capital One Chatbot Design 

# Let’s say we’re building a customer support chatbot for Capital One’s 
# online banking platform, and we have access to customer account details, 
# transaction histories, and common support FAQs.
# How would you approach designing this chatbot to ensure it provides secure, 
# helpful, and relevant responses to customers?

# store transaction histories and account details (structured data) in postgresql
# load customer support FAQs -> chunking -> add metadata -> metadata filtering
# When the query is related to transaction history or account details convert text
# to sql and fetch data from postgresql
# else retrieve from vectorstore and let LLM generate response