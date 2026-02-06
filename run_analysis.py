import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import numpy as np

# 대시보드 저장 경로 설정
OUTPUT_DIR = "/Users/yunjiho/ficb6/farminfo/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 폰트 설정 (Mac용)
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='Malgun Gothic')

plt.rc('axes', unicode_minus=False)

def load_and_preprocess_data(filepath):
    """데이터 로드 및 전처리"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # 날짜 변환
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'])
        df['주문월'] = df['주문일'].dt.to_period('M')
    
    # 숫자형 컬럼 변환 (콤마 제거)
    numeric_cols = ['결제금액', '판매단가', '공급단가', '주문취소 금액', '실결제 금액', '주문수량']
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            
    # 마진 계산
    if '판매단가' in df.columns and '공급단가' in df.columns:
        df['마진'] = (df['판매단가'] - df['공급단가']) * df.get('주문수량', 1)
    
    # [Simulation] 연령대 데이터 생성
    if '연령대' not in df.columns:
        print("Simulating Age Group data...")
        np.random.seed(42)
        age_groups = ['20대', '30대', '40대', '50대', '60대 이상']
        # 가중치: 3040 주축
        probs = [0.15, 0.30, 0.35, 0.15, 0.05] 
        df['연령대'] = np.random.choice(age_groups, size=len(df), p=probs)
        
    print(f"Data loaded: {len(df)} rows")
    return df

def analyze_sales(df):
    """1. 매출 및 수익성 분석"""
    print("\n--- 1. Sales Analysis ---")
    
    # 일별 매출 추이
    daily_sales = df.groupby(df['주문일'].dt.date)['실결제 금액'].sum()
    
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=daily_sales.index, y=daily_sales.values)
    plt.title('일별 매출 추이')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_daily_sales_trend.png'))
    plt.close()
    
    print(f"Total Sales: {df['실결제 금액'].sum():,.0f} KRW")
    if '마진' in df.columns:
        print(f"Average Margin per Order: {df['마진'].mean():,.0f} KRW")

def analyze_customers(df):
    """2. 고객 분석 (재구매 및 연령대)"""
    print("\n--- 2. Customer Analysis ---")
    
    # [재구매율]
    if '재구매 횟수' in df.columns:
        # 고객 단위 통계
        cust_stats = df.groupby('UID')['재구매 횟수'].max()
        repeat_customers = cust_stats[cust_stats > 0]
        repeat_rate = len(repeat_customers) / len(cust_stats) * 100
        print(f"Repurchase Rate: {repeat_rate:.2f}%")
        
        plt.figure(figsize=(6, 6))
        plt.pie([len(cust_stats)-len(repeat_customers), len(repeat_customers)], 
                labels=['신규', '재구매'], autopct='%1.1f%%', colors=['#E0E0E0','#FF8C00'])
        plt.title('고객 구성 (신규 vs 재구매)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '2_customer_repurchase.png'))
        plt.close()
    
    # [연령대 분석] - 시뮬레이션
    if '연령대' in df.columns:
        age_sales = df.groupby('연령대')['실결제 금액'].sum().sort_index()
        
        plt.figure(figsize=(8, 8))
        age_sales.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("OrRd", len(age_sales)))
        plt.title('연령대별 매출 비중')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '2_age_revenue_share.png'))
        plt.close()

def analyze_products(df):
    """3. 상품 분석"""
    print("\n--- 3. Product Analysis ---")
    
    # 상품별 판매량 (Top 10)
    top_products = df.groupby('상품명')['주문수량'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
    plt.title('Top 10 판매 상품 (주문수량 기준)')
    plt.xlabel('주문 수량')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_top_products.png'))
    plt.close()
    
    # 무게별 판매 비중
    if '무게 구분' in df.columns:
        weight_sales = df['무게 구분'].value_counts()
        plt.figure(figsize=(8, 8))
        weight_sales.plot(kind='pie', autopct='%1.1f%%')
        plt.title('무게별 판매 비중')
        plt.savefig(os.path.join(OUTPUT_DIR, '3_sales_by_weight.png'))
        plt.close()

def analyze_sellers(df):
    """4. 셀러 분석"""
    print("\n--- 4. Seller Analysis ---")
    if '셀러명' in df.columns:
        top_sellers = df.groupby('셀러명')['실결제 금액'].sum().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_sellers.values, y=top_sellers.index, palette='rocket')
        plt.title('Top 10 셀러 (매출 기준)')
        plt.xlabel('매출액')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_top_sellers.png'))
        plt.close()
    else:
        print("Seller column not found.")

def analyze_regions(df):
    """5. 지역 분석 (Bar Chart)"""
    print("\n--- 5. Regional Analysis ---")
    if '광역지역' in df.columns:
        region_sales = df.groupby('광역지역')['실결제 금액'].sum().sort_values()
        
        plt.figure(figsize=(10, 8))
        # 수평 막대 그래프
        sns.barplot(x=region_sales.values, y=region_sales.index, color='orange')
        plt.title('지역별 매출 규모')
        plt.xlabel('매출액')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_regional_sales.png'))
        plt.close()
    else:
        print("Region column not found.")

def main():
    # 경로 설정 주의 (Robust Path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "input", "preprocessed_data.csv")
    
    if not os.path.exists(input_path):
        # 만약 input 폴더가 현재 위치가 아니라면, 상위 혹은 하위 폴더 탐색
        input_path = "input/preprocessed_data.csv"
        
    df = load_and_preprocess_data(input_path)
    
    analyze_sales(df)
    analyze_customers(df)
    analyze_products(df)
    analyze_sellers(df)
    analyze_regions(df)
    
    print(f"\nAnalysis Complete. Charts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
