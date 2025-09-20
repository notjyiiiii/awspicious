# awspicious

## Social Media Auto Poster

A comprehensive Python system for automatically posting videos to Instagram, Twitter, and Facebook with scheduling capabilities.

### Features

- **Multi-platform posting**: Instagram, Twitter, and Facebook support
- **Video validation**: Automatic validation against platform specifications
- **Scheduling**: Schedule posts for specific times with recurring options
- **Bulk scheduling**: Schedule multiple videos from a folder
- **Web interface**: User-friendly web dashboard
- **CLI interface**: Command-line interface for automation
- **Error handling**: Comprehensive error handling and logging

### Setup

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure API Credentials

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Instagram/Facebook (Meta)
FACEBOOK_ACCESS_TOKEN=your_facebook_access_token
INSTAGRAM_BUSINESS_ACCOUNT_ID=your_instagram_business_account_id
FACEBOOK_PAGE_ID=your_facebook_page_id

# Twitter/X
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
```

### Usage

#### Web Interface

Start the web server:

```bash
cd backend
python web_interface.py
```

Visit `http://localhost:5000` in your browser.

#### Command Line Interface

Post immediately:
```bash
python backend/cli_interface.py post /path/to/video.mp4 --caption "Your caption" --platforms instagram twitter
```

Schedule a post:
```bash
python backend/cli_interface.py schedule /path/to/video.mp4 14:30 --caption "Scheduled post"
```

Start scheduler:
```bash
python backend/cli_interface.py start
```