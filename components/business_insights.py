# -*- coding: utf-8 -*-
"""
Hotel Business Intelligence System - Complete Fixed Version
Cung cấp insight toàn diện cho chủ khách sạn với visualizations chi tiết
"""

import pandas as pd
import numpy as np
import warnings
import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Optional/Heavy deps (guarded)
try:
    from pyvi import ViTokenizer  # type: ignore
except Exception:
    ViTokenizer = None  # type: ignore

try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # type: ignore

try:
    from wordcloud import WordCloud  # type: ignore
except Exception:
    WordCloud = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
except Exception:
    plt = None  # type: ignore
    sns = None  # type: ignore

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st

warnings.filterwarnings('ignore')
if plt is not None:
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
if sns is not None:
    sns.set_palette("husl")

# -----------------------------------------------------------------------------
# Simple lexicon-based sentiment fallback (VN + EN)
# -----------------------------------------------------------------------------
POSITIVE_WORDS = {
    'tốt','tuyệt','đẹp','sạch','thân thiện','hài lòng','xuất sắc','tuyệt vời',
    'good','great','nice','clean','friendly','amazing','wonderful','perfect','excellent','love'
}

NEGATIVE_WORDS = {
    'tệ','xấu','bẩn','ồn','ồn ào','không tốt','không sạch','không hài lòng','kém',
    'bad','dirty','noisy','terrible','awful','poor','worst','hate'
}

def _lexicon_sentiment_counts(texts):
    counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for t in texts:
        text = str(t).lower()
        pos = any(w in text for w in POSITIVE_WORDS)
        neg = any(w in text for w in NEGATIVE_WORDS)
        if pos and not neg:
            counts['Positive'] += 1
        elif neg and not pos:
            counts['Negative'] += 1
        elif pos and neg:
            counts['Neutral'] += 1
        else:
            counts['Neutral'] += 1
    return counts

# =============================================================================
# LAZY-LOAD SENTIMENT MODEL (faster startup on Streamlit Cloud)
# =============================================================================
sentiment_model = None  # global placeholder

@st.cache_resource(show_spinner=False)
def _load_hf_sentiment_model():
    if pipeline is None:
        return None
    try:
        return pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment")
    except Exception:
        try:
            return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
        except Exception:
            return None

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_hotel_data(info_path, comments_path):
    """Load hotel data from CSV"""
    hotel_info = pd.read_csv(info_path)
    hotel_comments = pd.read_csv(comments_path)

    hotel_info.columns = hotel_info.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
    hotel_comments.columns = hotel_comments.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

    print(f"✅ Loaded {len(hotel_info)} hotels and {len(hotel_comments)} reviews")
    return hotel_info, hotel_comments

def preprocess_data(hotel_info, hotel_comments):
    """Clean and preprocess hotel data"""
    score_columns = [
        'Total_Score', 'Location', 'Cleanliness', 'Service',
        'Facilities', 'Value_for_money', 'Comfort_and_room_quality'
    ]

    for col in score_columns:
        if col in hotel_info.columns:
            hotel_info[f"{col}_missing_flag"] = hotel_info[col].eq("No information").astype(int)
            hotel_info[col] = hotel_info[col].replace("No information", np.nan)
            if hotel_info[col].dtype == 'object':
                hotel_info[col] = hotel_info[col].str.replace(',', '.')
            hotel_info[col] = pd.to_numeric(hotel_info[col], errors='coerce')

    def extract_rank(rank_text):
        if pd.isna(rank_text):
            return 3.0
        match = re.search(r'(\d+)', str(rank_text))
        return float(match.group(1)) if match else 3.0

    if "Hotel_Rank" in hotel_info.columns:
        hotel_info['Hotel_Rank_numeric'] = hotel_info['Hotel_Rank'].apply(extract_rank)

    # Xử lý comments
    if "Score" in hotel_comments.columns:
        hotel_comments['Score'] = pd.to_numeric(hotel_comments['Score'], errors='coerce')
    if "Review_Date" in hotel_comments.columns:
        hotel_comments['Review_Date'] = pd.to_datetime(hotel_comments['Review_Date'], errors='coerce')
        hotel_comments['Year'] = hotel_comments['Review_Date'].dt.year
        hotel_comments['Month'] = hotel_comments['Review_Date'].dt.month
        hotel_comments['Quarter'] = hotel_comments['Review_Date'].dt.quarter

    # Tính toán system averages
    system_avg = {}
    for col in score_columns:
        if col in hotel_info.columns:
            system_avg[col.lower()] = hotel_info[col].mean()

    if 'comments_count' in hotel_info.columns:
        system_avg['comments_count'] = hotel_info['comments_count'].mean()
    if 'Hotel_Rank_numeric' in hotel_info.columns:
        system_avg['hotel_rank'] = hotel_info['Hotel_Rank_numeric'].mean()

    print(f"✅ Preprocessing done. System averages calculated for {len(system_avg)} metrics.")
    return hotel_info, hotel_comments, system_avg

# =============================================================================
# ADVANCED ANALYTICS ENGINE
# =============================================================================

class HotelAnalyticsEngine:
    """Advanced analytics for hotel business intelligence"""

    def __init__(self, hotel_info, hotel_comments, system_avg):
        self.hotel_info = hotel_info
        self.hotel_comments = hotel_comments
        self.system_avg = system_avg
        self.stop_words = {'và','là','có','không','được','cho','với','của','một','các',
                          'này','đó','rất','tôi','em','anh','chị','ạ','ở','về','đi','ra',
                          'vào','lên','xuống','khách','sạn','hotel','room','phòng'}

    def get_hotel_overview(self, hotel_id):
        """Thông tin tổng quan khách sạn"""
        hotel_data = self.hotel_info[self.hotel_info['Hotel_ID'] == hotel_id]
        if hotel_data.empty:
            return None

        hotel = hotel_data.iloc[0].to_dict()
        comments = self.hotel_comments[self.hotel_comments['Hotel_ID'] == hotel_id]

        overview = {
            'basic_info': {
                'name': hotel.get('Hotel_Name', 'Unknown'),
                'star_rating': hotel.get('Hotel_Rank', 'N/A'),
                'location': hotel.get('Address', 'N/A'),
                'total_score': hotel.get('Total_Score', 0)
            },
            'performance_summary': {
                'total_reviews': len(comments),
                'avg_score': comments['Score'].mean() if not comments.empty else 0,
                'latest_review': comments['Review_Date'].max() if not comments.empty else None,
                'review_trend': self._calculate_trend(comments)
            }
        }
        return overview

    def analyze_strengths_weaknesses(self, hotel_id):
        """Phân tích điểm mạnh & điểm yếu"""
        hotel_data = self.hotel_info[self.hotel_info['Hotel_ID'] == hotel_id]
        if hotel_data.empty:
            return None

        hotel = hotel_data.iloc[0]

        # So sánh với trung bình hệ thống
        metrics = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']
        strengths = []
        weaknesses = []

        for metric in metrics:
            if metric in hotel and not pd.isna(hotel[metric]):
                hotel_score = float(hotel[metric])
                system_avg = self.system_avg.get(metric.lower(), 7.0)

                diff = hotel_score - system_avg
                if diff > 0.3:
                    strengths.append({
                        'metric': metric,
                        'hotel_score': hotel_score,
                        'system_avg': system_avg,
                        'difference': diff
                    })
                elif diff < -0.3:
                    weaknesses.append({
                        'metric': metric,
                        'hotel_score': hotel_score,
                        'system_avg': system_avg,
                        'difference': diff
                    })

        return {
            'strengths': sorted(strengths, key=lambda x: x['difference'], reverse=True),
            'weaknesses': sorted(weaknesses, key=lambda x: x['difference'])
        }

    def analyze_customer_demographics(self, hotel_id):
        """Thống kê khách hàng"""
        comments = self.hotel_comments[self.hotel_comments['Hotel_ID'] == hotel_id]
        if comments.empty:
            return None

        # Phân tích theo thời gian
        time_analysis = {
            'by_quarter': comments.groupby('Quarter').size().to_dict() if 'Quarter' in comments else {},
            'by_month': comments.groupby('Month').size().to_dict() if 'Month' in comments else {},
            'by_year': comments.groupby('Year').size().to_dict() if 'Year' in comments else {}
        }

        # Phân tích điểm số
        score_distribution = comments['Score'].value_counts().sort_index().to_dict() if 'Score' in comments else {}

        return {
            'total_customers': len(comments),
            'time_trends': time_analysis,
            'score_distribution': score_distribution,
            'avg_score': comments['Score'].mean() if 'Score' in comments else 0
        }

    def extract_customer_insights(self, hotel_id):
        """Phân tích từ khóa trong nhận xét của khách hàng"""
        comments = self.hotel_comments[self.hotel_comments['Hotel_ID'] == hotel_id]
        if comments.empty or 'Body' not in comments.columns:
            return {'keywords': [], 'sentiment': {}, 'topics': []}

        # Xử lý text
        texts = []
        for text in comments['Body'].fillna(''):
            try:
                if ViTokenizer is not None:
                    segmented = ViTokenizer.tokenize(str(text).lower())
                else:
                    segmented = str(text).lower()
                words = re.findall(r'\b\w+\b', segmented)
                filtered = [w for w in words if len(w) > 2 and w not in self.stop_words]
                texts.append(' '.join(filtered))
            except Exception:
                texts.append(str(text).lower())

        # TF-IDF Keywords
        keywords = []
        try:
            if len([t for t in texts if t.strip()]) > 0:
                vectorizer = TfidfVectorizer(max_features=20, min_df=2, max_df=0.8, ngram_range=(1,2))
                tfidf_matrix = vectorizer.fit_transform([t for t in texts if t.strip()])
                feature_names = vectorizer.get_feature_names_out()
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                keywords = [(feature_names[i], mean_scores[i]) for i in range(len(feature_names))]
                keywords.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"⚠️ Keyword extraction error: {e}")

        # Sentiment Analysis (Hugging Face, batched, all reviews)
        sentiment_dist = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        try:
            all_texts = comments['Body'].dropna().astype(str).tolist() if 'Body' in comments.columns else []
            if all_texts:
                # Try cached HF model only if explicitly enabled (faster cold start)
                use_hf = st.session_state.get("bi_use_hf_sentiment", False)
                if use_hf:
                    global sentiment_model
                    if sentiment_model is None:
                        sentiment_model = _load_hf_sentiment_model()
                if sentiment_model is not None and use_hf:
                    batch_size = 64
                    for i in range(0, len(all_texts), batch_size):
                        chunk = all_texts[i:i+batch_size]
                        preds = sentiment_model(chunk, truncation=True)
                        for pred in preds:
                            label = str(pred.get('label', '')).upper()
                            if 'POS' in label or 'POSITIVE' in label or label == 'LABEL_2':
                                sentiment_dist['Positive'] += 1
                            elif 'NEG' in label or 'NEGATIVE' in label or label == 'LABEL_0':
                                sentiment_dist['Negative'] += 1
                            else:
                                sentiment_dist['Neutral'] += 1
                else:
                    # Fast lexicon method (default)
                    sentiment_dist = _lexicon_sentiment_counts(all_texts)
        except Exception as e:
            print(f"⚠️ Sentiment analysis error: {e}")

        return {
            'keywords': keywords[:15],
            'sentiment': sentiment_dist,
            'word_cloud_text': ' '.join([t for t in texts if t.strip()])
        }

    def benchmark_comparison(self, hotel_id):
        """So sánh với trung bình hệ thống"""
        hotel_data = self.hotel_info[self.hotel_info['Hotel_ID'] == hotel_id]
        if hotel_data.empty:
            return None

        hotel = hotel_data.iloc[0]

        comparison = {}
        metrics = ['Total_Score', 'Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']

        for metric in metrics:
            if metric in hotel and not pd.isna(hotel[metric]):
                hotel_score = float(hotel[metric])
                system_avg = self.system_avg.get(metric.lower(), 0)
                percentile = self._calculate_percentile(metric, hotel_score)

                comparison[metric] = {
                    'hotel_score': hotel_score,
                    'system_average': system_avg,
                    'difference': hotel_score - system_avg,
                    'percentile': percentile,
                    'performance': 'Excellent' if percentile >= 80 else 'Good' if percentile >= 60 else 'Average' if percentile >= 40 else 'Below Average'
                }

        return comparison

    def _calculate_trend(self, comments):
        """Tính toán xu hướng review"""
        if comments.empty or 'Review_Date' not in comments.columns:
            return 'No data'

        recent_comments = comments[comments['Review_Date'] >= (datetime.now() - timedelta(days=90))]
        old_comments = comments[comments['Review_Date'] < (datetime.now() - timedelta(days=90))]

        if recent_comments.empty or old_comments.empty:
            return 'Insufficient data'

        recent_avg = recent_comments['Score'].mean() if 'Score' in recent_comments else 0
        old_avg = old_comments['Score'].mean() if 'Score' in old_comments else 0

        diff = recent_avg - old_avg
        if diff > 0.2:
            return 'Improving'
        elif diff < -0.2:
            return 'Declining'
        else:
            return 'Stable'

    def _calculate_percentile(self, metric, score):
        """Tính percentile của hotel trong hệ thống"""
        if metric in self.hotel_info.columns:
            all_scores = pd.to_numeric(self.hotel_info[metric], errors='coerce').dropna()
            if len(all_scores) > 0:
                return (all_scores <= score).mean() * 100
        return 50  # Default

# =============================================================================
# COMPREHENSIVE VISUALIZATION DASHBOARD
# =============================================================================

class HotelVisualizationDashboard:
    """Comprehensive visualization dashboard"""

    def __init__(self, analytics_engine):
        self.engine = analytics_engine

    def create_executive_dashboard(self, hotel_id):
        """Tạo dashboard tổng quan cho executive"""
        overview = self.engine.get_hotel_overview(hotel_id)
        strengths_weak = self.engine.analyze_strengths_weaknesses(hotel_id)
        benchmark = self.engine.benchmark_comparison(hotel_id)
        customer_insights = self.engine.extract_customer_insights(hotel_id)

        if not overview:
            print("❌ Hotel not found!")
            return

        # Create comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Performance vs System Average", "Strengths & Weaknesses", "Sentiment Analysis",
                "Score Distribution", "Monthly Review Trends", "Key Performance Metrics",
                "Customer Satisfaction Trend", "Keyword Analysis", "Competitive Position"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]
            ]
        )

        # 1. Performance vs System Average
        if benchmark:
            metrics = list(benchmark.keys())
            hotel_scores = [benchmark[m]['hotel_score'] for m in metrics]
            system_avgs = [benchmark[m]['system_average'] for m in metrics]

            fig.add_trace(go.Bar(name="Hotel", x=metrics, y=hotel_scores, marker_color='lightblue'), row=1, col=1)
            fig.add_trace(go.Bar(name="System Avg", x=metrics, y=system_avgs, marker_color='orange'), row=1, col=1)

        # 2. Strengths & Weaknesses
        if strengths_weak:
            strengths = strengths_weak['strengths'][:3]
            weaknesses = strengths_weak['weaknesses'][:3]

            if strengths or weaknesses:
                all_items = [(s['metric'], s['difference'], 'Strength') for s in strengths] + \
                           [(w['metric'], abs(w['difference']), 'Weakness') for w in weaknesses]
                if all_items:
                    items, values, types = zip(*all_items)
                    colors = ['green' if t=='Strength' else 'red' for t in types]
                    fig.add_trace(go.Bar(x=list(items), y=list(values), marker_color=colors), row=1, col=2)

        # 3. Sentiment Distribution
        sentiment = customer_insights.get('sentiment', {})
        if sentiment and sum(sentiment.values()) > 0:
            fig.add_trace(go.Pie(labels=list(sentiment.keys()), values=list(sentiment.values())), row=1, col=3)

        # 4. Score Distribution
        demographics = self.engine.analyze_customer_demographics(hotel_id)
        if demographics and demographics['score_distribution']:
            scores = list(demographics['score_distribution'].keys())
            counts = list(demographics['score_distribution'].values())
            fig.add_trace(go.Histogram(x=scores, y=counts, marker_color='skyblue'), row=2, col=1)

        # 5. Monthly Trends
        if demographics and demographics['time_trends']['by_month']:
            months = list(demographics['time_trends']['by_month'].keys())
            review_counts = list(demographics['time_trends']['by_month'].values())
            fig.add_trace(go.Scatter(x=months, y=review_counts, mode='lines+markers'), row=2, col=2)

        # 6. KPI Indicator
        total_score = overview['basic_info']['total_score']
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=total_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Score"},
            delta={'reference': 8.0},
            gauge={'axis': {'range': [None, 10]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 5], 'color': "lightgray"},
                            {'range': [5, 8], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 9}}), row=2, col=3)

        # 7. Customer Satisfaction Trend (placeholder)
        fig.add_trace(go.Scatter(x=[1,2,3,4,5], y=[7,8,7.5,8.2,8.5],
                                mode='lines+markers', name="Satisfaction"), row=3, col=1)

        # 8. Top Keywords
        keywords = customer_insights.get('keywords', [])[:8]
        if keywords:
            words, scores = zip(*keywords)
            fig.add_trace(go.Bar(x=list(words), y=list(scores)), row=3, col=2)

        # 9. Competitive Position
        if benchmark:
            percentiles = [benchmark[m]['percentile'] for m in benchmark.keys()]
            metrics_names = list(benchmark.keys())
            fig.add_trace(go.Bar(x=metrics_names, y=percentiles, marker_color='gold'), row=3, col=3)

        fig.update_layout(
            title=f"🏨 Executive Dashboard: {overview['basic_info']['name']}",
            height=1200,
            width=1600,
            showlegend=True
        )

        fig.show()

        # Print Executive Summary
        print("="*80)
        print(f"🏨 EXECUTIVE SUMMARY: {overview['basic_info']['name']}")
        print("="*80)
        print(f"⭐ Overall Score: {total_score}/10")
        print(f"📊 Total Reviews: {overview['performance_summary']['total_reviews']}")
        print(f"📈 Trend: {overview['performance_summary']['review_trend']}")

        if strengths_weak:
            if strengths_weak['strengths']:
                print(f"💪 Top Strength: {strengths_weak['strengths'][0]['metric']}")
            if strengths_weak['weaknesses']:
                print(f"⚠️  Top Weakness: {strengths_weak['weaknesses'][0]['metric']}")

        sentiment_summary = max(sentiment.items(), key=lambda x: x[1]) if sentiment else ('Unknown', 0)
        print(f"😊 Dominant Sentiment: {sentiment_summary[0]} ({sentiment_summary[1]} reviews)")
        print("="*80)

    def create_detailed_analytics_report(self, hotel_id):
        """Tạo báo cáo phân tích chi tiết với dữ liệu thực"""
        customer_insights = self.engine.extract_customer_insights(hotel_id)
        demographics = self.engine.analyze_customer_demographics(hotel_id)

        # Create figure with better error handling
        if plt is None:
            st.warning("Matplotlib is not available. Skipping detailed analytics figure.")
            return
        plt.figure(figsize=(18, 12))

        # 1. Word Cloud
        plt.subplot(2, 4, 1)
        try:
            word_cloud_text = customer_insights.get('word_cloud_text', '')
            if WordCloud is not None and word_cloud_text and word_cloud_text.strip():
                wordcloud = WordCloud(width=400, height=300, background_color='white',
                                    max_words=50, colormap='viridis',
                                    prefer_horizontal=0.7).generate(word_cloud_text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title('Customer Keywords Cloud', fontsize=12, fontweight='bold')
                plt.axis('off')
            else:
                plt.text(0.5, 0.5, 'No text data\navailable', ha='center', va='center', fontsize=12)
                plt.title('Customer Keywords Cloud', fontsize=12, fontweight='bold')
                plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f'Word Cloud\nError: {str(e)[:20]}...', ha='center', va='center', fontsize=10)
            plt.title('Customer Keywords Cloud', fontsize=12, fontweight='bold')
            plt.axis('off')

        # 2. Sentiment Distribution
        plt.subplot(2, 4, 2)
        sentiment = customer_insights.get('sentiment', {})
        if sentiment and sum(sentiment.values()) > 0:
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            wedges, texts, autotexts = plt.pie(sentiment.values(), labels=sentiment.keys(),
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            plt.title('Sentiment Distribution', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No sentiment\ndata available', ha='center', va='center', fontsize=12)
            plt.title('Sentiment Distribution', fontsize=12, fontweight='bold')

        # 3. Top Keywords Bar Chart
        plt.subplot(2, 4, 3)
        keywords = customer_insights.get('keywords', [])[:8]
        if keywords:
            words = [w for w, _ in keywords]
            scores = [s for _, s in keywords]
            bars = plt.barh(range(len(words)), scores, color='lightcoral', alpha=0.8)
            plt.yticks(range(len(words)), words)
            plt.title('Top Keywords (TF-IDF)', fontsize=12, fontweight='bold')
            plt.xlabel('TF-IDF Score', fontsize=10)

            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                plt.text(score + 0.001, i, f'{score:.3f}', va='center', fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No keywords\nfound', ha='center', va='center', fontsize=12)
            plt.title('Top Keywords (TF-IDF)', fontsize=12, fontweight='bold')

        # 4. Score Distribution
        plt.subplot(2, 4, 4)
        if demographics and demographics['score_distribution']:
            scores = sorted(list(demographics['score_distribution'].keys()))
            counts = [demographics['score_distribution'][s] for s in scores]
            bars = plt.bar(scores, counts, color='skyblue', alpha=0.8, edgecolor='darkblue')
            plt.title('Score Distribution', fontsize=12, fontweight='bold')
            plt.xlabel('Score', fontsize=10)
            plt.ylabel('Count', fontsize=10)

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=9)
        else:
            plt.text(0.5, 0.5, 'No score\ndistribution data', ha='center', va='center', fontsize=12)
            plt.title('Score Distribution', fontsize=12, fontweight='bold')

        # 5. Monthly Trends
        plt.subplot(2, 4, 5)
        if demographics and demographics['time_trends']['by_month']:
            months = sorted(list(demographics['time_trends']['by_month'].keys()))
            counts = [demographics['time_trends']['by_month'][m] for m in months]
            plt.plot(months, counts, marker='o', linewidth=2, markersize=6, color='blue')
            plt.title('Monthly Review Trends', fontsize=12, fontweight='bold')
            plt.xlabel('Month', fontsize=10)
            plt.ylabel('Review Count', fontsize=10)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # Add value labels on points
            for x, y in zip(months, counts):
                plt.annotate(str(y), (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No monthly\ntrend data', ha='center', va='center', fontsize=12)
            plt.title('Monthly Review Trends', fontsize=12, fontweight='bold')

        # 6. Quarterly Analysis
        plt.subplot(2, 4, 6)
        if demographics and demographics['time_trends']['by_quarter']:
            quarters = sorted(list(demographics['time_trends']['by_quarter'].keys()))
            counts = [demographics['time_trends']['by_quarter'][q] for q in quarters]
            bars = plt.bar([f'Q{q}' for q in quarters], counts, color='orange', alpha=0.8)
            plt.title('Quarterly Review Distribution', fontsize=12, fontweight='bold')
            plt.xlabel('Quarter', fontsize=10)
            plt.ylabel('Review Count', fontsize=10)

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=9)
        else:
            plt.text(0.5, 0.5, 'No quarterly\ndata available', ha='center', va='center', fontsize=12)
            plt.title('Quarterly Review Distribution', fontsize=12, fontweight='bold')

        # 7. Review Length Analysis
        plt.subplot(2, 4, 7)
        comments = self.engine.hotel_comments[self.engine.hotel_comments['Hotel_ID'] == hotel_id]
        if not comments.empty and 'Body' in comments.columns:
            review_lengths = comments['Body'].fillna('').str.len()
            plt.hist(review_lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
            plt.title('Review Length Distribution', fontsize=12, fontweight='bold')
            plt.xlabel('Characters', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.axvline(review_lengths.mean(), color='red', linestyle='--',
                       label=f'Mean: {review_lengths.mean():.0f}')
            plt.legend(fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No review\nlength data', ha='center', va='center', fontsize=12)
            plt.title('Review Length Distribution', fontsize=12, fontweight='bold')

        # 8. Score vs Time Scatter
        plt.subplot(2, 4, 8)
        if not comments.empty and 'Score' in comments.columns and 'Review_Date' in comments.columns:
            valid_data = comments[comments['Review_Date'].notna() & comments['Score'].notna()]
            if not valid_data.empty:
                plt.scatter(valid_data['Review_Date'], valid_data['Score'],
                           alpha=0.6, color='purple', s=30)
                plt.title('Scores Over Time', fontsize=12, fontweight='bold')
                plt.xlabel('Date', fontsize=10)
                plt.ylabel('Score', fontsize=10)
                plt.xticks(rotation=45)

                # Add trend line
                if len(valid_data) > 1:
                    z = np.polyfit(range(len(valid_data)), valid_data['Score'], 1)
                    p = np.poly1d(z)
                    plt.plot(valid_data['Review_Date'], p(range(len(valid_data))),
                            "r--", alpha=0.8, linewidth=2)
            else:
                plt.text(0.5, 0.5, 'No valid\nscore/date data', ha='center', va='center', fontsize=12)
                plt.title('Scores Over Time', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No score/time\ndata available', ha='center', va='center', fontsize=12)
            plt.title('Scores Over Time', fontsize=12, fontweight='bold')

        plt.tight_layout(pad=2.0)
        plt.show()

    def create_review_timeline_analysis(self, hotel_id):
        """Tạo phân tích timeline chi tiết cho reviews - MISSING METHOD ADDED"""
        overview = self.engine.get_hotel_overview(hotel_id)
        if not overview:
            print("❌ Hotel not found!")
            return

        comments = self.engine.hotel_comments[self.engine.hotel_comments['Hotel_ID'] == hotel_id]
        if comments.empty:
            print("❌ No review data available for timeline analysis!")
            return

        # Create comprehensive timeline analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Review Volume Over Time", "Score Trends Over Time",
                "Seasonal Review Patterns", "Review Score Distribution by Period"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "box"}]
            ]
        )

        # Prepare data with better date handling
        valid_comments = comments[comments['Review_Date'].notna() & comments['Score'].notna()].copy()

        if not valid_comments.empty:
            valid_comments['YearMonth'] = valid_comments['Review_Date'].dt.to_period('M')
            valid_comments['Month_Name'] = valid_comments['Review_Date'].dt.month_name()

            # 1. Review Volume Over Time
            volume_data = valid_comments.groupby('YearMonth').size().reset_index(name='Count')
            volume_data['YearMonth_str'] = volume_data['YearMonth'].astype(str)

            fig.add_trace(
                go.Scatter(
                    x=volume_data['YearMonth_str'],
                    y=volume_data['Count'],
                    mode='lines+markers',
                    name='Review Volume',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )

            # 2. Score Trends Over Time
            score_trends = valid_comments.groupby('YearMonth')['Score'].mean().reset_index()
            score_trends['YearMonth_str'] = score_trends['YearMonth'].astype(str)

            fig.add_trace(
                go.Scatter(
                    x=score_trends['YearMonth_str'],
                    y=score_trends['Score'],
                    mode='lines+markers',
                    name='Average Score',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )

            # 3. Seasonal Patterns
            seasonal_data = valid_comments.groupby('Month_Name').size().reindex([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]).fillna(0)

            fig.add_trace(
                go.Bar(
                    x=seasonal_data.index,
                    y=seasonal_data.values,
                    name='Reviews by Month',
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )

            # 4. Score Distribution by Period
            if len(valid_comments) > 20:  # Only if enough data
                # Split into periods
                valid_comments_sorted = valid_comments.sort_values('Review_Date')
                mid_point = len(valid_comments_sorted) // 2

                early_scores = valid_comments_sorted.iloc[:mid_point]['Score']
                recent_scores = valid_comments_sorted.iloc[mid_point:]['Score']

                fig.add_trace(
                    go.Box(y=early_scores, name='Earlier Period', marker_color='lightblue'),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Box(y=recent_scores, name='Recent Period', marker_color='lightcoral'),
                    row=2, col=2
                )

        fig.update_layout(
            title=f"📅 Review Timeline Analysis: {overview['basic_info']['name']}",
            height=800,
            width=1400,
            showlegend=True
        )

        fig.show()

        # Print Timeline Summary
        if not valid_comments.empty:
            print("="*80)
            print(f"📅 TIMELINE ANALYSIS SUMMARY: {overview['basic_info']['name']}")
            print("="*80)
            print(f"📊 Total Reviews Analyzed: {len(valid_comments)}")
            print(f"📅 Date Range: {valid_comments['Review_Date'].min().strftime('%Y-%m-%d')} to {valid_comments['Review_Date'].max().strftime('%Y-%m-%d')}")

            # Monthly average
            avg_monthly = len(valid_comments) / len(valid_comments['YearMonth'].unique()) if len(valid_comments['YearMonth'].unique()) > 0 else 0
            print(f"📈 Average Reviews per Month: {avg_monthly:.1f}")

            # Best and worst months
            monthly_scores = valid_comments.groupby('Month_Name')['Score'].mean()
            if not monthly_scores.empty:
                best_month = monthly_scores.idxmax()
                worst_month = monthly_scores.idxmin()
                print(f"🌟 Best Month (by score): {best_month} ({monthly_scores[best_month]:.1f})")
                print(f"⚠️  Challenging Month: {worst_month} ({monthly_scores[worst_month]:.1f})")

            # Trend analysis
            if len(valid_comments) > 10:
                recent_period = valid_comments['Review_Date'] >= (valid_comments['Review_Date'].max() - pd.Timedelta(days=90))
                recent_avg = valid_comments[recent_period]['Score'].mean()
                overall_avg = valid_comments['Score'].mean()

                trend = "Improving" if recent_avg > overall_avg + 0.2 else "Declining" if recent_avg < overall_avg - 0.2 else "Stable"
                print(f"📈 Recent Trend (90 days): {trend} (Recent: {recent_avg:.1f} vs Overall: {overall_avg:.1f})")

            print("="*80)

# =============================================================================
# MAIN EXECUTION AND UTILITY FUNCTIONS
# =============================================================================

def main():
    """Main execution function"""
    # Đường dẫn file (cập nhật theo đường dẫn thực tế)
    info_path = "/content/drive/MyDrive/DL07_K306_PhamHongPhat_LeThiNgocPhuong/Project 2 - Recommendation System/2 - Data/hotel_info.csv"
    comments_path = "/content/drive/MyDrive/DL07_K306_PhamHongPhat_LeThiNgocPhuong/Project 2 - Recommendation System/2 - Data/hotel_comments.csv"

    try:
        # Load và preprocess data
        print("📊 Loading hotel data...")
        hotel_info, hotel_comments = load_hotel_data(info_path, comments_path)
        hotel_info, hotel_comments, system_avg = preprocess_data(hotel_info, hotel_comments)

        # Khởi tạo analytics engine
        analytics_engine = HotelAnalyticsEngine(hotel_info, hotel_comments, system_avg)
        dashboard = HotelVisualizationDashboard(analytics_engine)

        # Debug: Check available methods
        print("🔍 Available dashboard methods:")
        methods = [method for method in dir(dashboard) if not method.startswith('_')]
        for method in methods:
            print(f"   • {method}")
        print()

        # Chọn hotel để phân tích (có thể thay đổi hotel_id)
        sample_hotel_id = hotel_info['Hotel_ID'].iloc[0]
        print(f"🏨 Analyzing hotel: {sample_hotel_id}")

        # # ✅ In ra dataframe gốc của hotel_info và hotel_comments cho hotel_id
        # print("\n📄 HOTEL INFO (raw):")
        # print(hotel_info[hotel_info['Hotel_ID'] == sample_hotel_id])

        # print("\n💬 HOTEL COMMENTS (raw):")
        # print(hotel_comments[hotel_comments['Hotel_ID'] == sample_hotel_id])

        # Tạo Executive Dashboard
        print("\n📈 Creating Executive Dashboard...")
        dashboard.create_executive_dashboard(sample_hotel_id)

        # Tạo Detailed Analytics Report
        print("\n📋 Creating Detailed Analytics Report...")
        dashboard.create_detailed_analytics_report(sample_hotel_id)

        # ❌ ĐÃ XOÁ phần Review Timeline Analysis

        # ========================================================
        # THÊM IN RA REPORT DẠNG TEXT
        # ========================================================

        # Customer Demographics
        demographics = analytics_engine.analyze_customer_demographics(sample_hotel_id)
        if demographics:
            print("\nCUSTOMER DEMOGRAPHICS")
            print("-" * 50)
            # Top 5 Nationalities (nếu dữ liệu có cột Nationality)
            if "Nationality" in hotel_comments.columns:
                top_nationalities = hotel_comments[hotel_comments['Hotel_ID'] == sample_hotel_id]['Nationality'] \
                                    .value_counts().head(5)
                print("Top 5 Nationalities:")
                total_reviews = top_nationalities.sum()
                for i, (nat, count) in enumerate(top_nationalities.items(), start=1):
                    pct = count * 100 / total_reviews
                    print(f"  {i}. {nat}: {count} reviews ({pct:.1f}%)")

            # Customer Groups (nếu dữ liệu có cột Customer_Group)
            if "Customer_Group" in hotel_comments.columns:
                groups = hotel_comments[hotel_comments['Hotel_ID'] == sample_hotel_id]['Customer_Group'] \
                         .value_counts()
                print("\nCustomer Groups:")
                for group, count in groups.items():
                    pct = count * 100 / len(hotel_comments[hotel_comments['Hotel_ID'] == sample_hotel_id])
                    print(f"  {group}: {count} reviews ({pct:.1f}%)")

        # Performance Analysis
        strengths_weak = analytics_engine.analyze_strengths_weaknesses(sample_hotel_id)
        if strengths_weak:
            print("\nPERFORMANCE ANALYSIS")
            print("-" * 50)

            print("Top 3 Strengths:")
            for i, s in enumerate(strengths_weak['strengths'][:3], start=1):
                print(f"  {i}. {s['metric']}: {s['hotel_score']:.2f}/10 "
                      f"(Above Average, {s['difference']:+.2f} vs system)")

            print("\nAreas for Improvement:")
            for i, w in enumerate(strengths_weak['weaknesses'][:3], start=1):
                print(f"  {i}. {w['metric']}: {w['hotel_score']:.2f}/10 "
                      f"(Below Average, {w['difference']:+.2f} vs system)")

        # ========================================================
        # KẾT THÚC
        # ========================================================
        print("\n✅ Analysis completed successfully!")

    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()


def analyze_specific_hotel(hotel_id_input):
    """Phân tích hotel cụ thể theo ID"""
    info_path = "/content/drive/MyDrive/DL07_K306_PhamHongPhat_LeThiNgocPhuong/Project 2 - Recommendation System/2 - Data/hotel_info.csv"
    comments_path = "/content/drive/MyDrive/DL07_K306_PhamHongPhat_LeThiNgocPhuong/Project 2 - Recommendation System/2 - Data/hotel_comments.csv"

    hotel_info, hotel_comments = load_hotel_data(info_path, comments_path)
    hotel_info, hotel_comments, system_avg = preprocess_data(hotel_info, hotel_comments)

    analytics_engine = HotelAnalyticsEngine(hotel_info, hotel_comments, system_avg)
    dashboard = HotelVisualizationDashboard(analytics_engine)

    dashboard.create_executive_dashboard(hotel_id_input)
    dashboard.create_detailed_analytics_report(hotel_id_input)
    dashboard.create_review_timeline_analysis(hotel_id_input)

def compare_multiple_hotels(hotel_ids_list):
    """So sánh nhiều hotels cùng lúc"""
    info_path = "/content/drive/MyDrive/DL07_K306_PhamHongPhat_LeThiNgocPhuong/Project 2 - Recommendation System/2 - Data/hotel_info.csv"
    comments_path = "/content/drive/MyDrive/DL07_K306_PhamHongPhat_LeThiNgocPhuong/Project 2 - Recommendation System/2 - Data/hotel_comments.csv"

    hotel_info, hotel_comments = load_hotel_data(info_path, comments_path)
    hotel_info, hotel_comments, system_avg = preprocess_data(hotel_info, hotel_comments)

    analytics_engine = HotelAnalyticsEngine(hotel_info, hotel_comments, system_avg)

    # Create comparison dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Total Score Comparison", "Sentiment Comparison",
                       "Review Count Comparison", "Performance Radar"],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatterpolar"}]]
    )

    hotel_names = []
    total_scores = []
    sentiment_data = {}
    review_counts = []

    # Collect data for all hotels
    for hotel_id in hotel_ids_list:
        overview = analytics_engine.get_hotel_overview(hotel_id)
        if overview:
            hotel_names.append(overview['basic_info']['name'][:20] + "...")  # Truncate long names
            total_scores.append(float(overview['basic_info']['total_score']) if overview['basic_info']['total_score'] else 0)
            review_counts.append(overview['performance_summary']['total_reviews'])

            # Get sentiment data
            insights = analytics_engine.extract_customer_insights(hotel_id)
            sentiment = insights.get('sentiment', {})
            for sent_type in ['Positive', 'Negative', 'Neutral']:
                if sent_type not in sentiment_data:
                    sentiment_data[sent_type] = []
                sentiment_data[sent_type].append(sentiment.get(sent_type, 0))

    # Plot comparisons
    if hotel_names:
        # Total Scores
        fig.add_trace(go.Bar(x=hotel_names, y=total_scores, name="Total Score",
                            text=[f"{s:.1f}" for s in total_scores], textposition='auto'), row=1, col=1)

        # Sentiment Comparison
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        for i, (sent_type, values) in enumerate(sentiment_data.items()):
            fig.add_trace(go.Bar(x=hotel_names, y=values, name=sent_type,
                               marker_color=colors[i % 3]), row=1, col=2)

        # Review Counts
        fig.add_trace(go.Bar(x=hotel_names, y=review_counts, name="Review Count",
                            text=review_counts, textposition='auto'), row=2, col=1)

        # Performance Radar (first hotel only for demo)
        if hotel_ids_list:
            benchmark = analytics_engine.benchmark_comparison(hotel_ids_list[0])
            if benchmark:
                metrics = list(benchmark.keys())
                values = [benchmark[m]['hotel_score'] for m in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=hotel_names[0] if hotel_names else "Hotel"
                ), row=2, col=2)

    fig.update_layout(title="🏨 Multi-Hotel Comparison Dashboard", height=800, width=1400)
    fig.show()

def get_top_performing_hotels(top_n=10):
    """Tìm top hotels có performance tốt nhất"""
    info_path = "/content/drive/MyDrive/DL07_K306_PhamHongPhat_LeThiNgocPhuong/Project 2 - Recommendation System/2 - Data/hotel_info.csv"
    comments_path = "/content/drive/MyDrive/DL07_K306_PhamHongPhat_LeThiNgocPhuong/Project 2 - Recommendation System/2 - Data/hotel_comments.csv"

    hotel_info, hotel_comments = load_hotel_data(info_path, comments_path)
    hotel_info, hotel_comments, system_avg = preprocess_data(hotel_info, hotel_comments)

    # Filter valid hotels with scores
    valid_hotels = hotel_info[hotel_info['Total_Score'].notna() & (hotel_info['Total_Score'] > 0)]

    if valid_hotels.empty:
        print("❌ No valid hotels found with scores")
        return

    # Sort by Total_Score
    top_hotels = valid_hotels.nlargest(top_n, 'Total_Score')

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Top hotels by score
    plt.subplot(2, 2, 1)
    hotel_names = [name[:15] + "..." if len(name) > 15 else name for name in top_hotels['Hotel_Name']]
    scores = top_hotels['Total_Score']
    bars = plt.barh(range(len(hotel_names)), scores, color='gold', alpha=0.8)
    plt.yticks(range(len(hotel_names)), hotel_names)
    plt.xlabel('Total Score')
    plt.title(f'Top {top_n} Hotels by Score')
    plt.gca().invert_yaxis()

    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score + 0.1, i, f'{score:.1f}', va='center', fontweight='bold')

    # Score distribution of top hotels
    plt.subplot(2, 2, 2)
    plt.hist(top_hotels['Total_Score'], bins=8, alpha=0.7, color='lightblue', edgecolor='blue')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Hotels')
    plt.title('Score Distribution (Top Hotels)')
    plt.axvline(top_hotels['Total_Score'].mean(), color='red', linestyle='--',
               label=f'Mean: {top_hotels["Total_Score"].mean():.1f}')
    plt.legend()

    # Geographic distribution (if available)
    plt.subplot(2, 2, 3)
    if 'Address' in top_hotels.columns:
        # Extract cities from addresses (simple approach)
        cities = top_hotels['Address'].str.extract(r'([A-Za-z\s]+)')[0].fillna('Unknown')
        city_counts = cities.value_counts().head(5)
        plt.pie(city_counts.values, labels=city_counts.index, autopct='%1.1f%%')
        plt.title('Top Cities (Top Hotels)')
    else:
        plt.text(0.5, 0.5, 'No address\ndata available', ha='center', va='center')
        plt.title('Geographic Distribution')

    # Performance metrics comparison
    plt.subplot(2, 2, 4)
    metrics = ['Location', 'Cleanliness', 'Service', 'Facilities']
    avg_scores = []
    for metric in metrics:
        if metric in top_hotels.columns:
            avg_score = pd.to_numeric(top_hotels[metric], errors='coerce').mean()
            avg_scores.append(avg_score if not pd.isna(avg_score) else 0)
        else:
            avg_scores.append(0)

    bars = plt.bar(metrics, avg_scores, color='lightcoral', alpha=0.8)
    plt.ylabel('Average Score')
    plt.title('Average Metrics (Top Hotels)')
    plt.xticks(rotation=45)

    # Add score labels on bars
    for bar, score in zip(bars, avg_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print summary
    print("="*80)
    print(f"🏆 TOP {top_n} PERFORMING HOTELS")
    print("="*80)
    for i, (idx, hotel) in enumerate(top_hotels.iterrows()):
        print(f"{i+1:2d}. {hotel['Hotel_Name'][:50]:<50} | Score: {hotel['Total_Score']:.1f}")
    print("="*80)
    print(f"📊 Average Score: {top_hotels['Total_Score'].mean():.1f}")
    print(f"📈 Highest Score: {top_hotels['Total_Score'].max():.1f}")
    print(f"📉 Lowest Score: {top_hotels['Total_Score'].min():.1f}")
    print("="*80)

if __name__ == "__main__":
    main()

    # Để phân tích hotel cụ thể, uncomment dòng dưới và thay hotel_id
    # analyze_specific_hotel("YOUR_HOTEL_ID_HERE")


# =============================================================================
# STREAMLIT UI INTEGRATION FOR APP
# =============================================================================

@st.cache_data(show_spinner=False)
def _load_bi_data():
    """Load and preprocess business insight CSVs from hotel_ui_data/ with caching."""
    try:
        info_path = "hotel_ui_data/hotel_info.csv"
        comments_path = "hotel_ui_data/hotel_comments.csv"
        hotel_info, hotel_comments = load_hotel_data(info_path, comments_path)
        hotel_info, hotel_comments, system_avg = preprocess_data(hotel_info, hotel_comments)
        engine = HotelAnalyticsEngine(hotel_info, hotel_comments, system_avg)
        return engine, hotel_info, hotel_comments, system_avg
    except Exception as e:
        raise RuntimeError(f"Failed to load BI data: {e}")


def _render_performance_vs_avg(benchmark: dict):
    if not benchmark:
        return None
    metrics = list(benchmark.keys())
    hotel_scores = [benchmark[m]['hotel_score'] for m in metrics]
    system_avgs = [benchmark[m]['system_average'] for m in metrics]
    fig = go.Figure()
    fig.add_bar(name="Hotel", x=metrics, y=hotel_scores, marker_color='lightblue')
    fig.add_bar(name="System Avg", x=metrics, y=system_avgs, marker_color='orange')
    fig.update_layout(
        barmode='group',
        title_text="Performance vs System Average",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def _render_strengths_weaknesses(sw: dict):
    if not sw:
        return None
    strengths = sw.get('strengths', [])[:3]
    weaknesses = sw.get('weaknesses', [])[:3]
    items = []
    values = []
    colors = []
    for s in strengths:
        items.append(s['metric']); values.append(s['difference']); colors.append('green')
    for w in weaknesses:
        items.append(w['metric']); values.append(abs(w['difference'])); colors.append('red')
    if not items:
        return None
    fig = go.Figure(go.Bar(x=items, y=values, marker_color=colors))
    fig.update_layout(
        title_text="Top Strengths & Weaknesses",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def _render_sentiment(sentiment: dict):
    if not sentiment or sum(sentiment.values()) == 0:
        return None
    fig = go.Figure(go.Pie(labels=list(sentiment.keys()), values=list(sentiment.values())))
    fig.update_layout(
        title_text="Sentiment Distribution",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def _fallback_sentiment_from_scores(df_comments: pd.DataFrame) -> dict:
    if df_comments is None or df_comments.empty or 'Score' not in df_comments.columns:
        return {"Positive": 0, "Negative": 0, "Neutral": 0}
    scores = df_comments['Score'].dropna()
    return {
        "Positive": int((scores >= 8).sum()),
        "Negative": int((scores <= 5).sum()),
        "Neutral": int(((scores > 5) & (scores < 8)).sum()),
    }


def _render_score_distribution(demo: dict):
    if not demo or not demo.get('score_distribution'):
        return None
    scores = list(demo['score_distribution'].keys())
    counts = list(demo['score_distribution'].values())
    fig = go.Figure(go.Bar(x=scores, y=counts, marker_color='skyblue'))
    fig.update_layout(
        title_text="Score Distribution",
        height=300,
        xaxis_title="Score",
        yaxis_title="Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def _render_monthly_trend(demo: dict):
    if not demo or not demo.get('time_trends', {}).get('by_month'):
        return None
    months = list(demo['time_trends']['by_month'].keys())
    counts = list(demo['time_trends']['by_month'].values())
    fig = go.Figure(go.Scatter(x=months, y=counts, mode='lines+markers'))
    fig.update_layout(
        title_text="Monthly Review Trends",
        height=300,
        xaxis_title="Month",
        yaxis_title="Reviews",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def _render_pie_counts(series: pd.Series, title: str, top_n: int = 8):
    # Defensive: accept None or non-Series inputs
    if series is None:
        return None
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    s = s.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return None
    counts = s.value_counts()
    if counts.empty:
        return None
    if len(counts) > top_n:
        top = counts.head(top_n)
        other_sum = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other_sum})])
    fig = go.Figure(go.Pie(labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.3))
    fig.update_layout(
        title_text=title,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def render_business_insights():
    """Streamlit page: Business Insight. Allow choosing a hotel_id and show insights."""
    st.markdown("### 💼 Business Insight")
    st.caption("Chọn một Hotel ID để xem insight dành cho chủ khách sạn.")

    try:
        engine, hotel_info, hotel_comments, _ = _load_bi_data()
    except Exception as e:
        st.error(str(e))
        return

    # Toggle for heavy HF sentiment (default off for performance on Cloud)
    with st.expander("Advanced options", expanded=False):
        st.checkbox("Use Hugging Face sentiment (slower, more accurate)", key="bi_use_hf_sentiment", value=False)

    # Filters
    col_l, col_r = st.columns([2, 1])
    with col_l:
        hotel_name_query = st.text_input("Tìm theo tên khách sạn", "")
        df_filter = hotel_info
        if hotel_name_query:
            df_filter = hotel_info[hotel_info.get('Hotel_Name', '').astype(str).str.contains(hotel_name_query, case=False, na=False)]
        hotel_ids = df_filter['Hotel_ID'].tolist() if 'Hotel_ID' in df_filter.columns else []
        if not hotel_ids:
            st.warning("Không tìm thấy Hotel_ID phù hợp.")
            return
        hotel_id = st.selectbox("Hotel ID", hotel_ids, index=0)
    with col_r:
        st.metric("Tổng số khách sạn", f"{len(hotel_info):,}")
        st.metric("Tổng số reviews", f"{len(hotel_comments):,}")

    overview = engine.get_hotel_overview(hotel_id)
    if not overview:
        st.warning("Không tìm thấy thông tin khách sạn.")
        return

    strengths_weak = engine.analyze_strengths_weaknesses(hotel_id)
    benchmark = engine.benchmark_comparison(hotel_id)
    customer_ins = engine.extract_customer_insights(hotel_id)
    demographics = engine.analyze_customer_demographics(hotel_id)

    # Header info (removed Trend card)
    st.markdown(f"#### 🏨 {overview['basic_info'].get('name', 'Unknown')}")
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Overall Score", f"{overview['basic_info'].get('total_score', 0):.1f}/10")
    with kpi2:
        st.metric("Avg Review Score", f"{overview['performance_summary'].get('avg_score', 0):.2f}")
    with kpi3:
        st.metric("Total Reviews", f"{overview['performance_summary'].get('total_reviews', 0):,}")

    # Sentiment KPI row aligned with top metrics
    if customer_ins:
        sentiment_top = customer_ins.get('sentiment', {})
        if not sentiment_top or sum(sentiment_top.values()) == 0:
            sentiment_top = _fallback_sentiment_from_scores(hotel_comments[hotel_comments['Hotel_ID'] == hotel_id])
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            st.metric("Positive", str(sentiment_top.get("Positive", 0)))
        with sm2:
            st.metric("Neutral", str(sentiment_top.get("Neutral", 0)))
        with sm3:
            st.metric("Negative", str(sentiment_top.get("Negative", 0)))

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig1 = _render_performance_vs_avg(benchmark)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = _render_strengths_weaknesses(strengths_weak)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    # Sentiment pies + composition pies in one row (moved KPI cards above)
    if customer_ins:
        sentiment = customer_ins.get('sentiment', {})
        if not sentiment or sum(sentiment.values()) == 0:
            sentiment = _fallback_sentiment_from_scores(hotel_comments[hotel_comments['Hotel_ID'] == hotel_id])
        filtered_comments = hotel_comments[hotel_comments['Hotel_ID'] == hotel_id]
        fig_sent = _render_sentiment(sentiment)
        fig_nat = _render_pie_counts(filtered_comments.get('Nationality'), "Users by Nationality")
        fig_grp = _render_pie_counts(filtered_comments.get('Group_Name'), "Users by Group Name")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            if fig_sent:
                st.plotly_chart(fig_sent, use_container_width=True)
        with pc2:
            if fig_nat:
                st.plotly_chart(fig_nat, use_container_width=True)
        with pc3:
            if fig_grp:
                st.plotly_chart(fig_grp, use_container_width=True)

    # Distribution and trends
    col3, col4 = st.columns(2)
    with col3:
        fig4 = _render_score_distribution(demographics)
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)
    with col4:
        fig5 = _render_monthly_trend(demographics)
        if fig5:
            st.plotly_chart(fig5, use_container_width=True)

    # (Pie charts moved above with sentiment)