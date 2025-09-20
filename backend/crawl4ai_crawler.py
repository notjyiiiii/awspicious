import asyncio
from crawl4ai import AsyncWebCrawler
import json
from datetime import datetime
import os

class Crawl4AICrawler:
    def __init__(self):
        self.output_dir = "crawled_data"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    async def crawl_single_url(self, url):
        """Crawl a single URL using Crawl4AI"""
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(url=url)
            
            if result.success:
                return {
                    'url': url,
                    'title': result.metadata.get('title', ''),
                    'description': result.metadata.get('description', ''),
                    'content': result.markdown,
                    'links': result.links.get('internal', []) + result.links.get('external', []),
                    'crawl_time': datetime.now().isoformat(),
                    'status_code': result.status_code
                }
            else:
                print(f"Failed to crawl {url}: {result.error_message}")
                return None
    
    async def crawl_multiple_urls(self, urls):
        """Crawl multiple URLs concurrently"""
        async with AsyncWebCrawler(verbose=True) as crawler:
            results = []
            
            for url in urls:
                try:
                    result = await crawler.arun(url=url)
                    if result.success:
                        crawl_data = {
                            'url': url,
                            'title': result.metadata.get('title', ''),
                            'description': result.metadata.get('description', ''),
                            'content': result.markdown,
                            'links': result.links.get('internal', []) + result.links.get('external', []),
                            'crawl_time': datetime.now().isoformat(),
                            'status_code': result.status_code
                        }
                        results.append(crawl_data)
                        print(f"✅ Successfully crawled: {url}")
                    else:
                        print(f"❌ Failed to crawl {url}: {result.error_message}")
                except Exception as e:
                    print(f"❌ Error crawling {url}: {str(e)}")
            
            return results
    
    async def crawl_with_extraction(self, url, css_selector=None, extraction_strategy=None):
        """Crawl with specific content extraction"""
        async with AsyncWebCrawler(verbose=True) as crawler:
            # Configure extraction parameters
            extraction_config = {}
            if css_selector:
                extraction_config['css_selector'] = css_selector
            if extraction_strategy:
                extraction_config['extraction_strategy'] = extraction_strategy
            
            result = await crawler.arun(url=url, **extraction_config)
            
            if result.success:
                return {
                    'url': url,
                    'title': result.metadata.get('title', ''),
                    'extracted_content': result.extracted_content,
                    'markdown': result.markdown,
                    'links': result.links,
                    'crawl_time': datetime.now().isoformat()
                }
            else:
                print(f"Failed to crawl {url}: {result.error_message}")
                return None
    
    def save_data(self, data, filename=None):
        """Save crawled data to JSON file"""
        if not filename:
            filename = f"crawl4ai_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"✅ Data saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
    
    async def crawl_sitemap(self, sitemap_url):
        """Crawl URLs from a sitemap"""
        async with AsyncWebCrawler(verbose=True) as crawler:
            # First, crawl the sitemap to get URLs
            sitemap_result = await crawler.arun(url=sitemap_url)
            
            if not sitemap_result.success:
                print(f"Failed to fetch sitemap: {sitemap_result.error_message}")
                return []
            
            # Extract URLs from sitemap (this is a basic implementation)
            # You might need to parse XML properly for real sitemaps
            import re
            urls = re.findall(r'<loc>(.*?)</loc>', sitemap_result.html)
            
            print(f"Found {len(urls)} URLs in sitemap")
            
            # Crawl each URL
            results = await self.crawl_multiple_urls(urls[:10])  # Limit to first 10 for demo
            return results

# Example usage functions
async def example_single_url():
    """Example: Crawl a single URL"""
    crawler = Crawl4AICrawler()
    
    url = "https://example.com"  # Replace with your target URL
    result = await crawler.crawl_single_url(url)
    
    if result:
        crawler.save_data([result], "single_page_crawl.json")
        print(f"Crawled: {result['title']}")

async def example_multiple_urls():
    """Example: Crawl multiple URLs"""
    crawler = Crawl4AICrawler()
    
    urls = [
        "https://example.com",
        "https://example.com/about",
        "https://example.com/contact"
    ]  # Replace with your target URLs
    
    results = await crawler.crawl_multiple_urls(urls)
    crawler.save_data(results, "multiple_pages_crawl.json")
    print(f"Successfully crawled {len(results)} pages")

async def example_with_extraction():
    """Example: Crawl with specific content extraction"""
    crawler = Crawl4AICrawler()
    
    url = "https://example.com"  # Replace with your target URL
    
    # Extract only specific elements (e.g., articles, blog posts)
    result = await crawler.crawl_with_extraction(
        url=url,
        css_selector="article, .post, .content"  # Adjust based on target site structure
    )
    
    if result:
        crawler.save_data([result], "extracted_content.json")

async def main():
    """Main function - choose which example to run"""
    print("🚀 Starting Crawl4AI crawler...")
    
    # Uncomment the example you want to run:
    
    # await example_single_url()
    # await example_multiple_urls()
    # await example_with_extraction()
    
    # Or create your own crawling logic here
    crawler = Crawl4AICrawler()
    
    # Example: Crawl a single page
    url = input("Enter URL to crawl (or press Enter for example.com): ").strip()
    if not url:
        url = "https://example.com"
    
    result = await crawler.crawl_single_url(url)
    if result:
        crawler.save_data([result])
        print(f"\n✅ Successfully crawled: {result['title']}")
        print(f"📄 Content length: {len(result['content'])} characters")
        print(f"🔗 Found {len(result['links'])} links")

if __name__ == "__main__":
    asyncio.run(main())