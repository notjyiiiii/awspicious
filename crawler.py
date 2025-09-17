import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime
import os

class AIWebCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://example.com"  # Replace with your target URL
        self.output_dir = "crawled_data"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def make_request(self, url):
        """Make an HTTP request with error handling and rate limiting"""
        try:
            # Add delay to be respectful to the server
            time.sleep(2)
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def parse_content(self, html_content):
        """Parse the HTML content using BeautifulSoup"""
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        data = []
        
        try:
            # Modify these selectors based on the website structure
            articles = soup.find_all('article')
            
            for article in articles:
                item = {
                    'title': article.find('h2').text.strip() if article.find('h2') else '',
                    'content': article.find('div', class_='content').text.strip() if article.find('div', class_='content') else '',
                    'date': article.find('time').get('datetime') if article.find('time') else '',
                    'url': article.find('a')['href'] if article.find('a') else '',
                }
                data.append(item)
                
        except Exception as e:
            print(f"Error parsing content: {str(e)}")
        
        return data

    def save_data(self, data, filename=None):
        """Save the crawled data to a JSON file"""
        if not filename:
            filename = f"ai_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Data saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving data: {str(e)}")

    def crawl(self, num_pages=1):
        """Main crawling function"""
        all_data = []
        
        for page in range(1, num_pages + 1):
            print(f"Crawling page {page}...")
            
            # Modify this URL pattern according to the website's pagination structure
            url = f"{self.base_url}/page/{page}"
            
            html_content = self.make_request(url)
            if html_content:
                page_data = self.parse_content(html_content)
                all_data.extend(page_data)
            
        self.save_data(all_data)
        return all_data

def main():
    crawler = AIWebCrawler()
    # Crawl 5 pages (adjust as needed)
    data = crawler.crawl(num_pages=5)
    print(f"Total items crawled: {len(data)}")

if __name__ == "__main__":
    main()
