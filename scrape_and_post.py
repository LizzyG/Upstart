import os
import logging
import json
import openai
import yaml
import re
import requests
from dotenv import load_dotenv, find_dotenv
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin

# Load environment variables from .env file only if it exists
if find_dotenv():
    load_dotenv()

# Load environment variables for API keys
OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
CIRCLE_API_KEY = os.getenv("CIRCLE_API_KEY")
CIRCLE_COMMUNITY_ID = os.getenv("CIRCLE_COMMUNITY_ID")  # Community ID
CIRCLE_SPACE_ID = os.getenv("CIRCLE_SPACE_ID")  # Space ID
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"  # Enable dry run mode

# Set up OpenAI API
openai.api_key = OPENAI_API_KEY

# Circle API endpoint template
CIRCLE_API_ENDPOINT_TEMPLATE = "https://app.circle.so/api/v1/events/?community_id={community_id}&space_id={space_id}"

def load_config():
    """Load websites from the config file."""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['websites']

def scrape_website(url):
    """Scrape the website content using Playwright to handle JavaScript rendering and dynamic content."""
    logging.info(f"Scraping {url}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Visit the page
        page.goto(url)

        # Optionally, handle infinite scrolling
        logging.debug("Scrolling the page...")
        scroll_height = 1000
        while True:
            current_scroll_position = page.evaluate("window.scrollY + window.innerHeight")
            page.evaluate(f"window.scrollTo(0, {scroll_height});")
            scroll_height += 1000
            page.wait_for_timeout(1000)

            new_scroll_position = page.evaluate("window.scrollY + window.innerHeight")
            if new_scroll_position == current_scroll_position:
                break

        # Extract the HTML content after JavaScript execution
        html_content = page.content()

        browser.close()

    logging.info(f"Successfully scraped {url}")
    return html_content

def clean_html(html):
    """Clean the HTML by removing unnecessary tags, attributes, and focusing on the body content."""
    
    soup = BeautifulSoup(html, 'html.parser')

    # Extract the body content only
    body_content = soup.body

    if body_content is None:
        return ""  # If there's no body tag, return an empty string

    # Remove unnecessary tags (e.g., <script>, <style>, <svg>, <img>, <link>, <meta>, <noscript>)
    for tag in body_content(['script', 'style', 'svg', 'link', 'meta', 'noscript', 'img']):
        tag.decompose()  # Remove the tags from the soup

    # Remove comments
    comments = body_content.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # Remove all attributes from the remaining tags
    for tag in body_content.find_all(True):  # True means all tags
        if tag.name == 'a' and 'href' in tag.attrs:
            tag.attrs = {'href': tag.attrs['href']}  # Keep only href attribute
        else:
            tag.attrs = {}  # Clear all other attributes

    # Get the cleaned HTML content as a string
    cleaned_html = str(body_content)

    # Remove excessive whitespace (replace multiple spaces and line breaks with a single space)
    cleaned_html = re.sub(r'\s+', ' ', cleaned_html)

    # Optionally, format the result to be human-readable with minimal indentation
    cleaned_html = cleaned_html.strip()
    return cleaned_html


def extract_events_with_relevant_fields(html):
    """Use OpenAI to extract only relevant event fields."""

    soup = BeautifulSoup(html, 'html.parser')

    # Extract only the relevant parts of the HTML (e.g., event sections, titles, descriptions)
    cleaned_html = clean_html(html)
    # Define the function schema for structured output
    function_definition = {
        "name": "create_event",
        "description": "Extract and structure each individual event from the HTML as separate objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "events": {  # Multiple events should be returned as a list
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},  # Event Name
                            "body": {"type": "string"},  # Description
                            "link": {"type": "string"},  # Link to event details
                            "event_setting_attributes": {
                                "type": "object",
                                "properties": {
                                    "starts_at": {"type": "string", "format": "date-time"},  # Event Start
                                    "duration_in_seconds": {"type": "integer"},  # Duration of the event in seconds
                                    "location_type": {"type": "string"},  # Location type (e.g., virtual, in-person, tbd)
                                    "in_person_location": {"type": "string"}  # In-person location (if applicable)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    try:
        # Use the correct method from the new API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that extracts and structures event data from HTML."
                },
                {
                    "role": "user",
                    "content": f"Extract event details from the following HTML.  Be sure to capture all relevant information to help attendees understand what the event is.  If there is a url for more info about the event be sure to include it in the link field.  Be sure to list each event discretely:\n{cleaned_html}"
                }
            ],
            functions=[function_definition],
            function_call={"name": "create_event"},  # Forces the model to respond with structured data
            max_tokens=1500
        )
        # Access the function call arguments in the correct way
        function_call = response.choices[0].message.function_call

        # Extract the structured data from the function call arguments
        # Parse the arguments JSON string to a Python dictionary
        logging.debug(f"\nraw chatgpt response:  {function_call.arguments}")
        arguments = json.loads(function_call.arguments)
    except json.JSONDecodeError as e:
        # Handle JSON decoding errors and print useful debug info
        logging.error(f"Error parsing JSON response: {e}")
        logging.error(f"Raw response: {function_call.arguments}")
        return []

    # Ensure the structure is properly handled
    events = arguments.get('events', [])
    
    return events


def prepare_event_for_circle(event_data):
    """Fill in any missing fields to create a complete Circle API event structure."""
    body_text = event_data.get("body", "")
    body_with_link = f"{body_text}\n\n{event_data['link']}" if body_text else event_data["link"]
    
    complete_event = {
        "event": {
            "name": event_data.get("name", ""),
            "body": body_with_link,
            "slug": event_data.get("slug", ""),  # Optional slug
            "is_liking_disabled": False,
            "is_comments_disabled": False,
            "hide_meta_info": False,
            "hide_from_featured_areas": False,
            "event_setting_attributes": {
                "starts_at": event_data.get("event_setting_attributes", {}).get("starts_at", ""),  # Must be a valid datetime
                "duration_in_seconds": event_data.get("duration_in_seconds", 0),  # Default duration
                "location_type": event_data.get("location_type", "tbd"),  # Assume 'tbd' if missing
                "in_person_location": event_data.get("in_person_location", ""),
                "rsvp_disabled": False,
                "hide_location_from_non_attendees": False,
                "virtual_location_url": event_data.get("virtual_location_url", ""),
                "hide_attendees": False,
                "send_email_reminder": False,
                "send_in_app_notification_reminder": False
            },
            "meta_tag_attributes": {
                "meta_title": event_data.get("meta_title", ""),
                "meta_description": event_data.get("meta_description", ""),
                "opengraph_title": event_data.get("opengraph_title", ""),
                "opengraph_description": event_data.get("opengraph_description", ""),
                "opengraph_image": event_data.get("opengraph_image", "")
            }
        }
    }

    return complete_event


def post_event_to_circle(event_data):
    """Send the extracted events as structured JSON to Circle API."""
    
    # Build the API endpoint with community and space IDs
    api_url = CIRCLE_API_ENDPOINT_TEMPLATE.format(
        community_id=CIRCLE_COMMUNITY_ID,
        space_id=CIRCLE_SPACE_ID
    )
    
    if DRY_RUN:
        logging.info(f"\n[DRY RUN] Would send to Circle API at {api_url}")
        logging.info(f"Event Data:\n{json.dumps(event_data, indent=4)}")
    else:
        headers = {
            "Authorization": f"Token {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(api_url, json=event_data, headers=headers)

            if response.status_code == 201:
                logging.info(f"Successfully posted event: {event_data['event']['name']}")
            else:
                logging.error(f"Failed to post event. Status Code: {response.status_code}, Response: {response.text}")
        except requests.RequestException as e:
            logging.error(f"Error posting to Circle API: {e}")

class GitHubActionsHandler(logging.StreamHandler):
    """Custom handler to format logs for GitHub Actions."""
    def emit(self, record):
        log_entry = self.format(record)
        if record.levelno == logging.ERROR:
            print(f"::error::{log_entry}")
        elif record.levelno == logging.WARNING:
            print(f"::warning::{log_entry}")
        elif record.levelno == logging.INFO:
            print(f"::notice::{log_entry}")
        else:
            print(log_entry)


def main():
    """Main function to load config, scrape websites, extract events, and post to Circle API."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=getattr(logging, log_level, logging.INFO),  # Set log level from env or default
        handlers=[
            GitHubActionsHandler(),  # Use GitHub Actions handler for custom log levels
        ]
    )

    logging.info("Starting main process...")
    websites = load_config()
    
    for url in websites:
        html_content = scrape_website(url)
        
        if html_content:
            events_data = extract_events_with_relevant_fields(html_content)  # Get all relevant events as a list
            
            # Ensure there are events to process
            if not events_data:
                logging.warning(f"No events extracted from {url}")
                continue
            
            # Iterate over each event and post it to Circle API
            for event_data in events_data:
                event_data["link"] = urljoin(url, event_data.get("link", ""))
                complete_event = prepare_event_for_circle(event_data)  # Fill in missing fields
                logging.debug(f"Complete Event:\n{json.dumps(complete_event, indent=4)}")
                
                try:
                    post_event_to_circle(complete_event)
                except Exception as e:
                    logging.error(f"Failed to process event data: {e}")

if __name__ == "__main__":
    main()

