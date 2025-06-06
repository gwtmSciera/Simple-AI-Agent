from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from tools import (
    rating_summary_tool,
    search_reviews_tool,
    final_answer_tool,
    count_rating_tool,
    top_rated_comments_tool,
    low_rated_reasons_tool,
    review_count_by_date_tool,
    most_mentioned_dish_tool,
    sentiment_trend_tool,
    send_mail_tool,
)

app = FastAPI()

# Safe wrapper for tools to catch errors gracefully
def safe_tool(tool_func):
    def wrapper(input_str):
        try:
            return tool_func(input_str)
        except Exception as e:
            return f"[Tool Error] {str(e)}"
    return wrapper


### REVIEW AGENT SETUP ###

review_tools = [
    Tool(name="RatingSummary", func=safe_tool(rating_summary_tool), description="Summarize overall customer sentiment from all reviews. Input is an empty string."),
    Tool(name="SearchReviews", func=safe_tool(search_reviews_tool), description="Search reviews for specific keywords. Input is a query string."),
    Tool(name="FinalAnswer", func=safe_tool(final_answer_tool), description="Return the final answer to the user and stop the agent."),
    Tool(name="CountRating", func=safe_tool(count_rating_tool), description="Counts how many customers gave a specific rating. Input must be a string representing a number."),
    Tool(name="TopRatedComments", func=safe_tool(top_rated_comments_tool), description="Returns the top N reviews with a 5-star rating. Input is a number string like '3'."),
    Tool(name="LowRatedReasons", func=safe_tool(low_rated_reasons_tool), description="Returns common keywords found in 1-2 star reviews. Input is an empty string."),
    Tool(name="ReviewCountByDate", func=safe_tool(review_count_by_date_tool), description="Returns the number of reviews posted on a specific date. Input must be in YYYY-MM-DD format."),
    Tool(name="MostMentionedDish", func=safe_tool(most_mentioned_dish_tool), description="Returns the most frequently mentioned word (e.g., dish or term) in reviews."),
    Tool(name="SentimentTrend", func=safe_tool(sentiment_trend_tool), description="Returns the trend of average rating per month."),
]

llm_review = OllamaLLM(model="llama3.1")

tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in review_tools])

prompt_review = ZeroShotAgent.create_prompt(
    review_tools,
    prefix=(
        "You are an AI assistant that answers questions about restaurant reviews. "
        "You have access to the following tools:\n"
        f"{tool_descriptions}\n"
        "\nAlways think step-by-step and use the tools as needed. "
        "When you have enough information to answer, use the FinalAnswer tool."
    ),
    suffix=(
        "Begin!\n\n"
        "Question: {input}\n"
        "{agent_scratchpad}"
    ),
    input_variables=["input", "agent_scratchpad"],
)

llm_chain_review = LLMChain(llm=llm_review, prompt=prompt_review)

agent_review = ZeroShotAgent(llm_chain=llm_chain_review, tools=review_tools)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_review,
    tools=review_tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
)


### MAIL AGENT SETUP ###

# Mail agent tools
mail_tools = [
    Tool(name="SendMail", func=safe_tool(send_mail_tool), description="Send an email with given content. Input should be the email details as JSON string."),
    Tool(name="FinalAnswer", func=lambda x: x, description="Return the final answer to the user and stop the agent."),
]

llm_mail = OllamaLLM(model="llama3.1")

tool_descriptions_mail = "\n".join([f"- {tool.name}: {tool.description}" for tool in mail_tools])

prompt_mail = ZeroShotAgent.create_prompt(
    mail_tools,
    prefix=(
        "You are an AI assistant that helps to send emails.\n"
        "You have access to the following tools:\n"
        f"{tool_descriptions_mail}\n"
        "When you want to send an email, use the SendMail tool.\n"
        "For the SendMail tool input, provide ONLY a valid JSON string with the keys: "
        "`to` (email address), `subject` (email subject), and `body` (email content).\n"
        "Do NOT include any extra text, explanations, or formatting—only the JSON string.\n"
        "Always think step-by-step and use the tools as needed.\n"
        "When ready, use the FinalAnswer tool to finish.\n"
        "Once you have successfully sent an email, do not take any further action. Stop and return the final response immediately.\n"
        F"If you have successfully sent the email, stop further actions and provide a final answer: `Email sent successfully.` Do not resend the email again."
    ),
    suffix=(
        "Begin!\n\n"
        "Instruction: {input}\n"
        "{agent_scratchpad}"
    ),
    input_variables=["input", "agent_scratchpad"],
)

llm_chain_mail = LLMChain(llm=llm_mail, prompt=prompt_mail)

agent_mail = ZeroShotAgent(llm_chain=llm_chain_mail, tools=mail_tools)

mail_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_mail,
    tools=mail_tools,
    verbose=True,
    max_iterations=1,
    handle_parsing_errors=True,
)


### INTENT CLASSIFIER SETUP ###

intent_classifier_llm = OllamaLLM(model="llama3.1")

intent_prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "Classify the user's intent as one of the following categories: Review or Mail.\n\n"
        "User input: \"{user_input}\"\n\n"
        "Answer with exactly one word: Review or Mail."
    )
)

async def classify_intent_llm(user_input: str) -> str:
    prompt = intent_prompt_template.format(user_input=user_input)
    response = await intent_classifier_llm.apredict(prompt)
    intent = response.strip().capitalize()
    if intent not in ["Review", "Mail"]:
        intent = "Unknown"
    return intent


### API MODELS ###

class QuestionRequest(BaseModel):
    question: str

class SmartRequest(BaseModel):
    prompt: str


### API ENDPOINTS ###

@app.post("/ask")
async def ask_agent_endpoint(request: QuestionRequest):
    try:
        response = agent_executor.run(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sendmail")
async def send_mail_endpoint(request: QuestionRequest):
    try:
        response = mail_agent_executor.run(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/smart")
async def smart_router(request: SmartRequest):
    try:
        intent = await classify_intent_llm(request.prompt)
        if intent == "Review":
            result = agent_executor.run(request.prompt)
        elif intent == "Mail":
            result = mail_agent_executor.run(request.prompt)
        else:
            result = "Sorry, I couldn't understand if this is about reviews or sending mail."
        return {"intent": intent, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
