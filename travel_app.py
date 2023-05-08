import os

import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from datetime import datetime, timedelta
import replicate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.0)
image_prompt_prefix = "Dall-e Image Prompt:"

def generate_itinerary(place, days):
    """Generate itinerary using the langchain library and OpenAI's GPT-3 model."""
    prompt = PromptTemplate(
        input_variables=["place", "days"],
        template=""" 
        You are a holiday planner. I am visiting this {place} for {days} day(s). 
        I need to have only one suggestion for each day. 
        Plus return a creative Dall-e image prompt for the day 1 activity.

        Example 1:
        If there is only one day then return only one suggestion as below.
        Day 1: Visit the Eiffel Tower.
        Dall-e Image Prompt: A happy tourist poses in front of the Eiffel Tower.

        Example 2:
        If there are two days then return two suggestions as below.
        Day 1: Visit the Louvre Museum.
        Day 2: Visit the Eiffel Tower.
        Dall-e Image Prompt: A happy tourist poses in front of the Louvre Museum.
        """
    )
    story = LLMChain(llm=llm, prompt=prompt)
    return story.run(place=place, days=days)

def generate_images(prompt):
    """Generate images using the story text using the Replicate API."""
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": prompt}
    )
    return output

def app():
    st.title("Holiday Itinerary Generator")
    now = datetime.now()
    end_dt = now + timedelta(days=1)

    with st.form(key='my_form'):
        print("form")
        location_text = st.text_input(
            "Enter a location to visit",
            max_chars=None,
            value="New York",
            type="default",
            placeholder="Enter a location to visit",
        )
        start_date = st.date_input("Start date", now)
        end_date = st.date_input("End date", end_dt)

        if st.form_submit_button("Submit"):
            date_difference = end_date - start_date
            days = date_difference.days
            valid_days = days > 0 and days < 8
            valid_form = location_text and start_date and end_date
            if not valid_form:
                st.info("Please enter a location and dates.")
            if not valid_days:
                st.info("Please enter a duration of 1 to 7 days.")

            if valid_form and valid_days:
                with st.spinner('Generating itinerary...'):
                    itinerary = generate_itinerary(location_text, days)
                    text = itinerary.split(image_prompt_prefix)
                    st.write(text[0])
                    images = generate_images(text[1])
                    for item in images:
                        st.image(item)
                    
                    st.write(f"{image_prompt_prefix} {text[1]}")

if __name__ == '__main__':
    app()
