#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive HTML Report Generator

This script generates an interactive HTML report from Amazon product review summaries,
creating a multi-page interface with category images and clickable elements to view
detailed information about top and bottom-rated products.
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import base64
from io import BytesIO

# Default image URLs for each category (replace with actual images)
DEFAULT_CATEGORY_IMAGES = {
    "Accessories": "https://images-na.ssl-images-amazon.com/images/G/01/AmazonBasics/landing/electronics._CB485921693_.jpg",
    "Tablets & Entertainment": "https://images-na.ssl-images-amazon.com/images/G/01/kindle/journeys/YTNiNWIyZTgt/YTNiNWIyZTgt-ZjZmMzY2Yjct-w1500._CB417267304_.jpg",
    "Smart Home & Speakers": "https://images-na.ssl-images-amazon.com/images/G/01/kindle/journeys/Nzg3NzIxZDct/Nzg3NzIxZDct-YzA3MzI3Yjgt-w1500._CB418667506_.jpg",
    "E-readers": "https://images-na.ssl-images-amazon.com/images/G/01/kindle/journeys/Yzg5NWM0MDQt/Yzg5NWM0MDQt-YTJmMDQzMWIt-w1500._CB418667506_.jpg"
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate an interactive HTML report from Amazon review summaries')
    parser.add_argument('--input', type=str, required=True, help='Path to the CSV file with ChatGPT summaries')
    parser.add_argument('--output_dir', type=str, default='report', help='Directory to save the HTML report')
    parser.add_argument('--title', type=str, default='Amazon Product Analysis', help='Title for the report')
    return parser.parse_args()

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create subdirectories for assets
    assets_dir = os.path.join(output_dir, 'assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"Created assets directory: {assets_dir}")
    
    css_dir = os.path.join(assets_dir, 'css')
    if not os.path.exists(css_dir):
        os.makedirs(css_dir)
    
    js_dir = os.path.join(assets_dir, 'js')
    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
    
    img_dir = os.path.join(assets_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    return assets_dir, css_dir, js_dir, img_dir

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def analyze_category_data(df, category):
    """Analyze data for a specific category and generate statistics."""
    category_df = df[df['product_category'] == category].copy()
    
    if len(category_df) == 0:
        return None
    
    # Get top 3 products by average rating
    product_ratings = category_df.groupby('name')['reviews.rating'].mean().reset_index()
    top_products = product_ratings.sort_values('reviews.rating', ascending=False).head(3)
    
    # Get worst product
    worst_product = product_ratings.sort_values('reviews.rating').head(1)
    
    # Get positive summaries for top products
    top_products_data = []
    for _, product in top_products.iterrows():
        product_name = product['name']
        product_reviews = category_df[category_df['name'] == product_name]
        
        positive_reviews = product_reviews[product_reviews['rating_sentiment'] == 'Positive']
        if len(positive_reviews) > 0:
            # Get most representative positive summaries
            positive_summaries = positive_reviews['chatgpt_summary'].dropna().tolist()[:5]
        else:
            positive_summaries = []
        
        top_products_data.append({
            'name': product_name,
            'rating': product['reviews.rating'],
            'positive_summaries': positive_summaries
        })
    
    # Get improvement suggestions for worst product
    worst_product_data = None
    if not worst_product.empty:
        product_name = worst_product.iloc[0]['name']
        product_reviews = category_df[category_df['name'] == product_name]
        
        negative_reviews = product_reviews[product_reviews['rating_sentiment'] == 'Negative']
        if len(negative_reviews) > 0:
            # Get improvement suggestions from negative summaries
            improvement_summaries = negative_reviews['chatgpt_summary'].dropna().tolist()[:5]
        else:
            improvement_summaries = []
        
        worst_product_data = {
            'name': product_name,
            'rating': worst_product.iloc[0]['reviews.rating'],
            'improvement_summaries': improvement_summaries
        }
    
    # Generate sentiment distribution chart
    sentiment_counts = category_df['rating_sentiment'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', ax=ax, color=sns.color_palette('viridis', 3))
    plt.title(f'Sentiment Distribution - {category}', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    
    sentiment_chart = fig_to_base64(fig)
    plt.close(fig)
    
    # Generate rating distribution chart
    fig, ax = plt.subplots(figsize=(8, 5))
    category_df['reviews.rating'].value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue')
    plt.title(f'Rating Distribution - {category}', fontsize=14)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    
    rating_chart = fig_to_base64(fig)
    plt.close(fig)
    
    return {
        'category': category,
        'product_count': len(category_df['name'].unique()),
        'review_count': len(category_df),
        'avg_rating': category_df['reviews.rating'].mean(),
        'top_products': top_products_data,
        'worst_product': worst_product_data,
        'sentiment_chart': sentiment_chart,
        'rating_chart': rating_chart,
        'image_url': DEFAULT_CATEGORY_IMAGES.get(category, '')
    }

def create_css_file(css_dir):
    """Create CSS file for styling the HTML report."""
    css_content = """
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
        margin: 0;
        padding: 0;
        background-color: #f8f9fa;
    }
    
    .container {
        width: 90%;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    header {
        background-color: #232f3e;
        color: white;
        padding: 1rem 0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    h1, h2, h3, h4 {
        font-weight: 600;
    }
    
    .category-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-bottom: 2rem;
    }
    
    .category-card {
        background-color: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }
    
    .category-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    
    .category-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
    }
    
    .category-info {
        padding: 1.5rem;
    }
    
    .category-name {
        font-size: 1.5rem;
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #232f3e;
    }
    
    .category-stats {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .stat {
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #232f3e;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #666;
    }
    
    .category-detail {
        display: none;
        background-color: white;
        border-radius: 8px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .detail-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .back-button {
        background-color: #232f3e;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9rem;
    }
    
    .detail-section {
        margin-bottom: 2rem;
    }
    
    .product-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #232f3e;
    }
    
    .product-name {
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    .product-rating {
        color: #ff9900;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .summary-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .summary-item {
        background-color: white;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .chart-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 2rem;
    }
    
    .chart-box {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .chart-image {
        width: 100%;
        height: auto;
    }
    
    .improvement-card {
        background-color: #fff8e1;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff9900;
    }
    
    @media (max-width: 768px) {
        .category-grid {
            grid-template-columns: 1fr;
        }
        
        .chart-container {
            grid-template-columns: 1fr;
        }
    }
    """
    
    with open(os.path.join(css_dir, 'style.css'), 'w') as f:
        f.write(css_content)

def create_js_file(js_dir):
    """Create JavaScript file for interactive functionality."""
    js_content = """
    document.addEventListener('DOMContentLoaded', function() {
        // Get all category cards
        const categoryCards = document.querySelectorAll('.category-card');
        const categoryDetails = document.querySelectorAll('.category-detail');
        const backButtons = document.querySelectorAll('.back-button');
        const categoryGrid = document.querySelector('.category-grid');
        
        // Add click event to each category card
        categoryCards.forEach(card => {
            card.addEventListener('click', function() {
                const categoryId = this.getAttribute('data-category');
                
                // Hide category grid
                categoryGrid.style.display = 'none';
                
                // Show corresponding detail section
                document.getElementById(`detail-${categoryId}`).style.display = 'block';
                
                // Scroll to top
                window.scrollTo(0, 0);
            });
        });
        
        // Add click event to back buttons
        backButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Hide all detail sections
                categoryDetails.forEach(detail => {
                    detail.style.display = 'none';
                });
                
                // Show category grid
                categoryGrid.style.display = 'grid';
                
                // Scroll to top
                window.scrollTo(0, 0);
            });
        });
    });
    """
    
    with open(os.path.join(js_dir, 'script.js'), 'w') as f:
        f.write(js_content)

def generate_html_report(df, output_dir, title):
    """Generate HTML report from the dataframe."""
    # Create output directories
    assets_dir, css_dir, js_dir, img_dir = create_output_directory(output_dir)
    
    # Create CSS and JS files
    create_css_file(css_dir)
    create_js_file(js_dir)
    
    # Get unique categories
    categories = df['product_category'].unique()
    
    # Analyze data for each category
    category_data = []
    for category in categories:
        data = analyze_category_data(df, category)
        if data:
            category_data.append(data)
    
    # Create HTML template
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ title }}</title>
        <link rel="stylesheet" href="assets/css/style.css">
    </head>
    <body>
        <header>
            <div class="container">
                <h1>{{ title }}</h1>
                <p>Interactive report of Amazon product reviews by category</p>
            </div>
        </header>
        
        <div class="container">
            <!-- Category Grid -->
            <div class="category-grid">
                {% for category in categories %}
                <div class="category-card" data-category="{{ loop.index }}">
                    <img src="{{ category.image_url }}" alt="{{ category.category }}" class="category-image">
                    <div class="category-info">
                        <h2 class="category-name">{{ category.category }}</h2>
                        <div class="category-stats">
                            <div class="stat">
                                <div class="stat-value">{{ category.product_count }}</div>
                                <div class="stat-label">Products</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">{{ category.review_count }}</div>
                                <div class="stat-label">Reviews</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">{{ "%.1f"|format(category.avg_rating) }}</div>
                                <div class="stat-label">Avg Rating</div>
                            </div>
                        </div>
                        <p>Click to view detailed analysis</p>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <!-- Category Detail Pages -->
            {% for category in categories %}
            <div id="detail-{{ loop.index }}" class="category-detail">
                <div class="detail-header">
                    <h2>{{ category.category }} Analysis</h2>
                    <button class="back-button">← Back to Categories</button>
                </div>
                
                <div class="detail-section">
                    <h3>Top 3 Products</h3>
                    {% for product in category.top_products %}
                    <div class="product-card">
                        <h4 class="product-name">{{ product.name }}</h4>
                        <div class="product-rating">★ {{ "%.1f"|format(product.rating) }}</div>
                        
                        {% if product.positive_summaries %}
                        <h5>What Customers Love:</h5>
                        <ul class="summary-list">
                            {% for summary in product.positive_summaries %}
                            <li class="summary-item">{{ summary }}</li>
                            {% endfor %}
                        </ul>
                        {% else %}
                        <p>No positive summaries available.</p>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                
                {% if category.worst_product %}
                <div class="detail-section">
                    <h3>Areas for Improvement</h3>
                    <div class="improvement-card">
                        <h4 class="product-name">{{ category.worst_product.name }}</h4>
                        <div class="product-rating">★ {{ "%.1f"|format(category.worst_product.rating) }}</div>
                        
                        {% if category.worst_product.improvement_summaries %}
                        <h5>Suggested Improvements:</h5>
                        <ul class="summary-list">
                            {% for summary in category.worst_product.improvement_summaries %}
                            <li class="summary-item">{{ summary }}</li>
                            {% endfor %}
                        </ul>
                        {% else %}
                        <p>No improvement suggestions available.</p>
                        {% endif %}
                    </div>
                </div>
                {% endif %}
                
                <div class="chart-container">
                    <div class="chart-box">
                        <h3>Sentiment Distribution</h3>
                        <img src="data:image/png;base64,{{ category.sentiment_chart }}" alt="Sentiment Distribution" class="chart-image">
                    </div>
                    <div class="chart-box">
                        <h3>Rating Distribution</h3>
                        <img src="data:image/png;base64,{{ category.rating_chart }}" alt="Rating Distribution" class="chart-image">
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <script src="assets/js/script.js"></script>
    </body>
    </html>
    """
    
    # Render template
    template = Environment(loader=FileSystemLoader('.')).from_string(html_template)
    html_content = template.render(title=title, categories=category_data)
    
    # Write HTML file
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated successfully at {os.path.join(output_dir, 'index.html')}")

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Check if required columns exist
    required_columns = ['name', 'product_category', 'reviews.rating', 'rating_sentiment', 'chatgpt_summary']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
        print("Please make sure your input CSV contains these columns.")
        return
    
    # Generate HTML report
    print(f"Generating HTML report with title: {args.title}")
    generate_html_report(df, args.output_dir, args.title)
    
    print("Done!")

if __name__ == "__main__":
    main()
