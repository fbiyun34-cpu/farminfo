import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 환경 설정 및 비밀키 관리 (Secret Management)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Farminfo Analytics",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 비밀키 로드 함수 (Local vs Cloud Hybrid)
def get_naver_api_secrets():
    """
    Naver API 키를 로드합니다.
    1순위: Streamlit Cloud Secrets (st.secrets)
    2순위: 로컬 .env 파일
    """
    # 1. Streamlit Cloud Secrets 확인
    # 1. Streamlit Cloud Secrets 확인
    try:
        if "naver_api" in st.secrets:
            return st.secrets["naver_api"]["client_id"], st.secrets["naver_api"]["client_secret"]
    except FileNotFoundError:
        pass # secrets.toml이 없는 경우 무시하고 로컬 .env 시도
    except Exception:
        pass # 기타 에러 무시
    
    # 2. 로컬 .env 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # .env 후보 경로
    env_candidates = [
        os.path.join(project_root, ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    
    for env_path in env_candidates:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
            
    c_id = os.getenv('NAVER_CLIENT_ID')
    c_secret = os.getenv('NAVER_CLIENT_SECRET')
    
    if c_id and c_secret:
        return c_id, c_secret
    
    return None, None

client_id, client_secret = get_naver_api_secrets()

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 (Data Loading)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """데이터를 로드하고 캐싱합니다."""
    # 파일 경로 (절대 경로 또는 상대 경로)
    # ---------------------------
    # [Path Debugging Strategy]
    # ---------------------------
    # Streamlit Cloud와 로컬 환경의 경로 차이를 해결하기 위한 후보군 탐색
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../output
    project_root = os.path.dirname(current_dir)             # .../farminfo
    
    candidate_paths = [
        # 1. 스크립트 기준 상대 경로 (로컬/Cloud 일반적)
        os.path.join(project_root, "input", "preprocessed_data.csv"),
        # 2. 현재 작업 디렉토리(CWD) 기준 입수 (Streamlit Cloud Root 실행 시)
        os.path.join(os.getcwd(), "input", "preprocessed_data.csv"),
        # 3. Mount 경로 하드코딩 (최후의 수단, 리포지토리명에 따라 다를 수 있음)
        "/mount/src/farminfo/input/preprocessed_data.csv", 
        "input/preprocessed_data.csv"
    ]
    
    filepath = None
    for path in candidate_paths:
        if os.path.exists(path):
            filepath = path
            break
            
    if filepath is None:
        st.error("🚨 데이터 파일을 찾을 수 없습니다.")
        st.write("### Debug Info")
        st.write(f"- Current Working Dir: `{os.getcwd()}`")
        st.write(f"- Script Loc: `{current_dir}`")
        st.write("#### Checked Paths:")
        for p in candidate_paths:
            st.write(f"- `{p}`")
            
        # 디렉토리 구조 힌트 제공
        st.write("#### Directory Structure (Root):")
        try:
            st.write(os.listdir(os.getcwd()))
            if os.path.exists("input"):
                 st.write(f"input dir contents: {os.listdir('input')}")
        except Exception as e:
            st.write(f"Error listing dir: {e}")
            
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    
    # 날짜 변환
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'])
        df['주문월'] = df['주문일'].dt.to_period('M').astype(str)
        df['주문시간'] = df['주문일'].dt.hour
        df['요일'] = df['주문일'].dt.day_name()
    
    # [Simulation] 연령대 데이터 생성 (데모용)
    if '연령대' not in df.columns:
        import numpy as np
        # 재현성을 위해 시드 고정
        np.random.seed(42)
        age_groups = ['20대', '30대', '40대', '50대', '60대 이상']
        # 가중치: 3040 주축
        probs = [0.15, 0.30, 0.35, 0.15, 0.05] 
        df['연령대'] = np.random.choice(age_groups, size=len(df), p=probs)
    
    # 숫자형 컬럼 변환 (콤마 제거)
    numeric_cols = ['결제금액', '판매단가', '공급단가', '주문취소 금액', '실결제 금액', '주문수량']
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # 마진 계산
    if '판매단가' in df.columns and '공급단가' in df.columns:
        df['마진'] = (df['판매단가'] - df['공급단가']) * df.get('주문수량', 1)
        
    return df

raw_df = load_data()

if raw_df.empty:
    st.stop()

# -----------------------------------------------------------------------------
# 3. 사이드바 및 프롬프트 (Sidebar & Prompt UI)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ 컨트롤 패널")
    
    # 기간 설정
    min_date = raw_df['주문일'].min().date()
    max_date = raw_df['주문일'].max().date()
    
    date_range = st.date_input(
        "기간 선택",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # 빠른 필터
    st.divider()
    all_channels = raw_df['주문경로'].unique().tolist()
    selected_channels = st.multiselect("주문 경로 필터", all_channels, default=all_channels)
    
    if '이벤트 여부' in raw_df.columns:
        show_event_only = st.checkbox("이벤트 주문만 보기")
    else:
        show_event_only = False
        
    st.info(f"Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    
    # API 상태 표시 (보안상 키 자체는 노출 X)
    if client_id:
        st.success("Naver API Key Loaded ✅")
    else:
        st.warning("Naver API Key Not Found ⚠️")

# 메인 프롬프트 영역
st.markdown("## 🍊 Farminfo Prompt Analytics")
prompt = st.text_input(
    "분석하고 싶은 키워드를 입력하세요 (예: 서울, 감귤, 선물, 카카오톡)", 
    placeholder="키워드를 입력하면 관련 데이터만 필터링하여 깊이 있게 분석합니다.",
    help="상품명, 옵션, 주소, 주문경로 등에서 키워드를 검색합니다."
)

# -----------------------------------------------------------------------------
# 4. 데이터 필터링 로직 (Filtering Logic)
# -----------------------------------------------------------------------------
df_filtered = raw_df.copy()

# 1. 기간 필터
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_filtered[
        (df_filtered['주문일'].dt.date >= start_date) & 
        (df_filtered['주문일'].dt.date <= end_date)
    ]

# 2. 채널 필터
if selected_channels:
    df_filtered = df_filtered[df_filtered['주문경로'].isin(selected_channels)]

# 3. 이벤트 필터
if show_event_only and '이벤트 여부' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['이벤트 여부'] == 'Y']

# 4. 프롬프트(검색어) 필터 - 핵심 로직
if prompt:
    with st.spinner(f"'{prompt}' 관련 데이터 분석 중..."):
        # 검색 대상 컬럼
        search_cols = ['상품명', '옵션코드', '주소', '주문경로', '목적', '고객선택옵션']
        valid_cols = [c for c in search_cols if c in df_filtered.columns]
        
        # 키워드 포함 여부 마스크 생성 (OR 조건)
        mask = pd.Series(False, index=df_filtered.index)
        for col in valid_cols:
            mask |= df_filtered[col].astype(str).str.contains(prompt, case=False)
        
        df_filtered = df_filtered[mask]
        
        if df_filtered.empty:
            st.warning(f"'{prompt}'에 대한 검색 결과가 없습니다.")
            st.stop()
        else:
            st.success(f"'{prompt}' 키워드로 {len(df_filtered):,}건의 데이터를 찾았습니다.")

# -----------------------------------------------------------------------------
# 5. KPI 메트릭 (Metrics) [Table Like 1]
# -----------------------------------------------------------------------------
total_sales = df_filtered['실결제 금액'].sum()
total_orders = len(df_filtered)
avg_order_value = total_sales / total_orders if total_orders > 0 else 0
avg_margin = df_filtered['마진'].mean() if '마진' in df_filtered.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 매출액", f"{total_sales:,.0f}원")
col2.metric("총 주문수", f"{total_orders:,}건")
col3.metric("평균 객단가 (AOV)", f"{avg_order_value:,.0f}원")
col4.metric("평균 마진", f"{avg_margin:,.0f}원")

st.divider()

# -----------------------------------------------------------------------------
# 5. 메인 대시보드 구조 (Sidebar Navigation)
# -----------------------------------------------------------------------------

# 사이드바 메뉴 구성
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "분석 메뉴 선택",
    [
        "📊 매출 및 성과 (Sales)",
        "🍊 상품 분석 (Product)",
        "📢 채널 및 지역 (Ch & Reg)",
        "👥 고객 분석 (Customer)",
        "📈 셀러 분석 (Seller)"
    ]
)

st.title(f"{menu}")
st.markdown("---")

# -----------------------------------------------------------
# View 1: 📊 매출 및 성과 Analysis
# -----------------------------------------------------------
if "매출" in menu:
    # [View 1] 📊 매출 및 성과 Analysis
    
    # [Row 1] KPI 지표
    total_sales = df_filtered['실결제 금액'].sum()
    total_orders = df_filtered['주문수량'].sum()
    unique_customers = df_filtered['주문자명'].nunique()
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("총 매출액", f"{total_sales:,.0f} 원")
    col_kpi2.metric("총 주문수량", f"{total_orders:,.0f} 개")
    col_kpi3.metric("순수 구매자 수", f"{unique_customers:,.0f} 명")
    
    st.divider()

    # [수익화 제안] 목표 객단가 (Target AOV) Analysis
    st.subheader("💡 수익화 제안: 목표 객단가 (Target AOV)")
    
    # 마진 데이터 가용성 확인
    has_margin = '마진' in df_filtered.columns
    
    if has_margin and not df_filtered.empty:
        # 상위 25% 고마진 주문 기준
        threshold = df_filtered['마진'].quantile(0.75)
        high_margin_orders = df_filtered[df_filtered['마진'] >= threshold]
        
        if not high_margin_orders.empty:
            current_aov = total_sales / total_orders if total_orders > 0 else 0
            target_aov = high_margin_orders['실결제 금액'].mean()
            
            # 목표가 현재보다 낮으면 (고마진 상품이 저가일 경우) 상위 10% 매출 주문으로 대체 로직
            if target_aov <= current_aov:
                 target_aov = df_filtered['실결제 금액'].quantile(0.90) # 상위 10% 금액 기준

            upside_potential = (target_aov - current_aov) * total_orders
            
            p_col1, p_col2 = st.columns([1, 2])
            
            with p_col1:
                st.metric(
                    label="목표 객단가 (Target AOV)", 
                    value=f"{target_aov:,.0f} 원", 
                    delta=f"+{target_aov - current_aov:,.0f} 원",
                    help="상위 25% 고마진 주문들의 평균 객단가입니다."
                )
                st.caption(f"현재 AOV: {current_aov:,.0f} 원")
                
            with p_col2:
                st.info(f"""
                **💰 수익 극대화 전략**
                
                객단가를 **{target_aov:,.0f}원**으로 높인다면, 
                총 매출이 약 **{upside_potential:,.0f}원** 증가할 수 있습니다.
                
                **추천 액션:**
                - {target_aov:,.0f}원 이상 구매 시 **무료 배송** 또는 **사은품** 증정
                - **세트 상품(번들)** 구성을 통해 주문 금액 상향 유도
                - 장바구니 페이지에서 **추가 옵션(Cross-sell)** 노출 강화
                """)
                
            # [Bundle Proposal]
            st.markdown("#### 🎁 추천 상품 구성 (Golden Bundle)")
            
            # 1. Anchor Product (가장 많이 팔린 상품)
            top_anchor = df_filtered.groupby('상품명').agg({'주문수량':'sum', '실결제 금액':'mean', '마진':'mean'}).reset_index()
            anchor_row = top_anchor.sort_values('주문수량', ascending=False).iloc[0]
            anchor_name = anchor_row['상품명']
            anchor_price = anchor_row['실결제 금액']
            
            # 2. Add-on Product (Target AOV를 맞추기 위한 고마진 상품)
            gap_to_target = max(0, target_aov - anchor_price)
            
            # 후보군: Anchor가 아니면서, 평균 가격이 Gap 이상인 상품 중 마진이 가장 높은 것
            candidates = top_anchor[top_anchor['상품명'] != anchor_name]
            addon_candidates = candidates[candidates['실결제 금액'] >= gap_to_target * 0.8] # 갭의 80% 이상 커버 가능한 상품
            
            if not addon_candidates.empty:
                addon_row = addon_candidates.sort_values('마진', ascending=False).iloc[0]
                addon_name = addon_row['상품명']
                addon_price = addon_row['실결제 금액']
                bundle_price = anchor_price + addon_price
                
                # [Targeting Analysis]
                # 제안된 두 상품(Anchor, Add-on)을 구매한 이력 분석
                target_products = [anchor_name, addon_name]
                target_df = df_filtered[df_filtered['상품명'].isin(target_products)]
                
                if not target_df.empty:
                    # 1. Top Region
                    if '광역지역' in target_df.columns:
                        top_region = target_df['광역지역'].value_counts().idxmax()
                    else:
                        top_region = "전국"
                        
                    # 2. Top Age Group
                    if '연령대' in target_df.columns:
                        top_age = target_df['연령대'].value_counts().idxmax()
                    else:
                        top_age = "전 연령"
                        
                    targeting_info = f"\n\n**🎯 타겟 마케팅 (Target Audience):**\n- **추천 지역:** {top_region}\n- **핵심 연령:** {top_age} 고객층"
                else:
                    targeting_info = ""

                st.success(f"""
                **추천 번들: {anchor_name} + {addon_name}**
                
                - **구성:** {anchor_name} ({anchor_price:,.0f}원) + {addon_name} ({addon_price:,.0f}원)
                - **번들 가격:** {bundle_price:,.0f}원 (목표 객단가 {target_aov:,.0f}원 상회 🚀)
                - **기대 효과:** 고객이 가장 선호하는 상품에 고마진 상품을 제안하여 객단가와 이익 동반 상승
                {targeting_info}
                """)
            else:
                st.write("Target AOV를 달성할 적절한 추가 상품을 찾지 못했습니다.")

        else:
            st.info("고마진 주문 데이터가 충분하지 않아 목표를 산출할 수 없습니다.")
    else:
        st.warning("마진 데이터(공급단가/판매단가)가 없어 수익성 분석을 수행할 수 없습니다.")

    st.divider()
    
    # [Graph 1] 일별 매출 추이
    st.subheader("📈 일별 매출 및 주문 추이")
    daily_sales = df_filtered.groupby('주문일').agg({'실결제 금액':'sum', '주문수량':'sum'}).reset_index()
    
    fig_daily = px.line(daily_sales, x='주문일', y='실결제 금액', title='일별 매출 추이')
    fig_daily.update_traces(line_color='#FF9F40', line_width=3)
    st.plotly_chart(fig_daily, use_container_width=True)

elif "상품" in menu:
    # [View 2] 🍊 상품 분석 Analysis
    st.header("🍊 상품 분석 (Product Analysis)")
    
    col_prod1, col_prod2 = st.columns([1, 1])
    
    with col_prod1:
        st.subheader("🍩 카테고리별 판매 비중")
        if '품종' in df_filtered.columns and '무게 구분' in df_filtered.columns:
            fig_sun = px.sunburst(
                df_filtered, 
                path=['품종', '무게 구분'], 
                values='실결제 금액',
                color='실결제 금액',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.warning("품종/무게 데이터가 없습니다.")
            
    with col_prod2:
        st.subheader("🏆 상품 판매 순위 (Top 10)")
        top_products = df_filtered.groupby('상품명')['실결제 금액'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_bar = px.bar(top_products, x='실결제 금액', y='상품명', orientation='h', text_auto='.2s')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        fig_bar.update_traces(marker_color='#FF8C00')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # [Advanced Product Analysis]
    st.subheader("🔬 심층 상품 분석 (Advanced Product Analysis)")

    # Data Preparation per Product
    if not df_filtered.empty:
        prod_stats = df_filtered.groupby('상품명').agg({
            '실결제 금액': 'sum', 
            '주문수량': 'sum',
            '주문자명': 'nunique' # Number of Buyers
        }).reset_index()
        
        # Calculate Margin if available
        if '마진' in df_filtered.columns:
            margin_sum = df_filtered.groupby('상품명')['마진'].sum().reset_index()
            prod_stats = prod_stats.merge(margin_sum, on='상품명', how='left')
        else:
            prod_stats['마진'] = 0
            
        # 1. ABC Analysis (Pareto)
        prod_stats = prod_stats.sort_values('실결제 금액', ascending=False)
        prod_stats['Cumulative Sales'] = prod_stats['실결제 금액'].cumsum()
        prod_stats['Cumulative Perc'] = prod_stats['Cumulative Sales'] / prod_stats['실결제 금액'].sum()
        
        def assign_grade(row):
            if row['Cumulative Perc'] <= 0.8: return 'A (핵심)'
            elif row['Cumulative Perc'] <= 0.95: return 'B (일반)'
            else: return 'C (부진)'
            
        prod_stats['Grade'] = prod_stats.apply(assign_grade, axis=1)
        
        # Summary of Grades
        grade_counts = prod_stats['Grade'].value_counts().sort_index()
        
        st.markdown("##### 1. ABC 등급 분석 (Pareto Principle)")
        st.caption("매출 기여도 상위 80%를 A등급, 차위 15%를 B등급, 하위 5%를 C등급으로 분류합니다.")
        
        col_abc1, col_abc2 = st.columns([1, 2])
        
        with col_abc1:
            st.write("**등급별 상품 수**")
            st.dataframe(grade_counts, use_container_width=True)
            
        with col_abc2:
            fig_pareto = px.bar(
                prod_stats.head(20), 
                x='상품명', 
                y='실결제 금액',
                color='Grade',
                title='Top 20 상품 매출 기여도',
                color_discrete_map={'A (핵심)': '#E74C3C', 'B (일반)': '#F1C40F', 'C (부진)': '#95A5A6'}
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

        st.divider()
        
        # 2. Product Portfolio Map (Treemap)
        st.markdown("##### 2. 상품 포트폴리오 맵 (Treemap)")
        st.caption("계층: 등급 > 상품명 | 크기: 매출액 | 색상: 마진율 (초록색=고수익, 붉은색=저수익)")
        
        # Treemap requires non-negative values for size. Filter out negative sales.
        tree_df = prod_stats[prod_stats['실결제 금액'] > 0].copy()
        
        # Calculate Margin Rate for Color
        tree_df['마진율'] = (tree_df['마진'] / tree_df['실결제 금액']) * 100
        tree_df['마진율'] = tree_df['마진율'].fillna(0)
        
        # Format for Hover
        tree_df['매출액_fmt'] = tree_df['실결제 금액'].apply(lambda x: f"{x:,.0f}원")
        tree_df['마진_fmt'] = tree_df['마진'].apply(lambda x: f"{x:,.0f}원")
        tree_df['마진율_fmt'] = tree_df['마진율'].apply(lambda x: f"{x:.1f}%")
        tree_df['주문수_fmt'] = tree_df['주문수량'].apply(lambda x: f"{x:,.0f}개")

        fig_treemap = px.treemap(
            tree_df,
            path=[px.Constant("전체 상품"), 'Grade', '상품명'],
            values='실결제 금액',
            color='마진율',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=tree_df['마진율'].median(), # 中間값 기준
            custom_data=['매출액_fmt', '마진_fmt', '마진율_fmt', '주문수_fmt'],
            title="상품 계층별 매출 및 수익성(마진율) 분석"
        )
        
        fig_treemap.update_traces(
            textinfo="label+value+percent entry",
            hovertemplate="<b>%{label}</b><br>매출: %{customdata[0]}<br>마진: %{customdata[1]}<br>마진율: %{customdata[2]}<br>주문수: %{customdata[3]}"
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # 3. Detailed Data Table
        st.markdown("##### 3. 상품별 상세 지표")
        
        # Add basic formatting
        display_df = prod_stats.copy()
        display_df['실결제 금액'] = display_df['실결제 금액'].apply(lambda x: f"{x:,.0f}")
        display_df['마진'] = display_df['마진'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            display_df[['Grade', '상품명', '실결제 금액', '주문수량', '주문자명', '마진', 'Cumulative Perc']],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("분석할 상품 데이터가 없습니다.")

elif "채널" in menu:
    # [View 3] 📢 채널 및 지역 Analysis
    st.header("📢 채널 및 지역 분석 (Channel & Region)")
    
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        st.subheader("📢 주문 경로(채널) 효율")
        channel_perf = df_filtered.groupby('주문경로')[['실결제 금액', '주문수량']].sum().reset_index()
        fig_ch = px.bar(channel_perf, x='주문경로', y='실결제 금액', color='주문경로', title="채널별 매출액")
        st.plotly_chart(fig_ch, use_container_width=True)
        
    with col_ch2:
        st.subheader("📍 지역별 매출 규모")
        if '광역지역' in df_filtered.columns:
            region_stats = df_filtered.groupby('광역지역')['실결제 금액'].sum().reset_index().sort_values('실결제 금액', ascending=True)
            fig_bar_region = px.bar(
                region_stats,
                x='실결제 금액',
                y='광역지역',
                orientation='h',
                text_auto='.2s',
                title="지역별 매출액"
            )
            fig_bar_region.update_traces(marker_color='#FF8C00')
            st.plotly_chart(fig_bar_region, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------
    # [Regional Expansion Strategy]
    # -----------------------------------------------------------
    st.subheader("🗺️ 지역 확장 전략 (Regional Expansion Strategy)")
    
    if '광역지역' in df_filtered.columns:
        # 1. Target Region Selector
        all_regions = df_filtered['광역지역'].unique().tolist()
        # Default to the region with highest sales if available, else standard sort
        default_region = region_stats.iloc[-1]['광역지역'] if not region_stats.empty else all_regions[0]
        
        target_region = st.selectbox(
            "공략할 지역을 선택하세요",
            all_regions,
            index=all_regions.index(default_region) if default_region in all_regions else 0
        )
        
        # Filter for Target Region
        region_df = df_filtered[df_filtered['광역지역'] == target_region]
        
        if not region_df.empty:
            # Expert Analysis Data Prep
            total_sales_all = df_filtered['실결제 금액'].sum()
            current_region_sales = region_df['실결제 금액'].sum()
            region_share = current_region_sales / total_sales_all if total_sales_all > 0 else 0
            
            # 1. Market Classification
            if region_share >= 0.10: # 점유율 10% 이상은 핵심 지역
                region_type = "👑 핵심 거점 (Core Market)"
                strategy_focus = "충성도 강화 & 객단가 상승 (Lock-in & Up-sell)"
                growth_rate = 0.15 # 이미 성숙한 시장은 목표 성장률을 조금 낮게 잡음
            else:
                region_type = "🌱 성장 잠재 지역 (Growth Market)"
                strategy_focus = "신규 고객 확보 & 인지도 확대 (Acquisition)"
                growth_rate = 0.30 # 성장 초기 지역은 공격적인 목표 설정

            # 2. Demographics & Channel
            dominant_age = region_df['연령대'].value_counts().idxmax() if '연령대' in region_df.columns else "알 수 없음"
            dominant_channel = region_df['주문경로'].value_counts().idxmax()

            # 3. Top Products & Revenue Projection
            top3_products = region_df.groupby('상품명')['실결제 금액'].sum().nlargest(3).reset_index()
            potential_sales = current_region_sales * (1 + growth_rate)
            upside = potential_sales - current_region_sales
            
            # UI Layout
            strat_col1, strat_col2 = st.columns([1, 2])
            
            with strat_col1:
                st.markdown(f"#### 📊 지역 위상 및 목표")
                st.info(f"**{region_type}**\n\n매출 비중: **{region_share*100:.1f}%**")
                
                st.metric(
                    "현재 매출", 
                    f"{current_region_sales:,.0f} 원"
                )
                st.metric(
                    f"목표 매출 (+{growth_rate*100:.0f}%)",
                    f"{potential_sales:,.0f} 원",
                    delta=f"+{upside:,.0f} 원"
                )
            
            with strat_col2:
                st.markdown(f"#### 전략 리포트")
                st.caption(f"🎯 타겟 페르소나: **{dominant_age}** | 📢 최적 채널: **{dominant_channel}**")
                
                # Dynamic Recommendations
                st.markdown(f"**전략 초점: {strategy_focus}**")
                
                rec_list = []
                top_prod_name = top3_products.iloc[0]['상품명']
                
                if "핵심" in region_type:
                    rec_list.append(f"**VIP 마케팅**: {target_region} 내 구매 이력 보유 고객에게 **시크릿 쿠폰** 발송 (재구매 유도)")
                    rec_list.append(f"**번들링 강화**: 1위 상품인 '{top_prod_name}' 구매 시, 다른 상품 합배송 할인 제안 (객단가 UP)")
                    rec_list.append(f"**채널 최적화**: {dominant_channel} 채널의 충성 고객 대상으로 멤버십 혜택 혹은 정기 배송 안내")
                else:
                    rec_list.append(f"**공격적 침투**: {dominant_channel} 광고 예산을 {target_region} 지역에 집중 집행")
                    rec_list.append(f"**미끼 상품 전략**: '{top_prod_name}'의 소용량/체험팩을 기획하여 진입 장벽 낮추기")
                    rec_list.append(f"**로컬 타겟팅**: {target_region} 맘카페/커뮤니티 제휴를 통해 '{target_region} 한정 무료 배송' 이벤트 홍보")

                for i, rec in enumerate(rec_list, 1):
                    st.write(f"{i}. {rec}")
                    
                st.markdown("---")
                st.write(f"**🏆 {target_region} Best 3**")
                cols = st.columns(3)
                for idx, row in top3_products.iterrows():
                    with cols[idx]:
                        st.caption(f"{idx+1}위")
                        st.write(f"**{row['상품명']}**")
                        st.caption(f"{row['실결제 금액']:,.0f}원")

            st.markdown("---")
            
            # [Age Group Strategy Analysis]
            st.subheader(f"👥 {target_region} 연령별 공략 전략")
            
            if '연령대' in region_df.columns:
                # 1. Age Distribution Chart
                age_dist = region_df.groupby('연령대')['실결제 금액'].sum().reset_index()
                
                age_col1, age_col2 = st.columns([1, 1])
                
                with age_col1:
                     fig_age_donut = px.pie(
                        age_dist, 
                        values='실결제 금액', 
                        names='연령대', 
                        hole=0.4,
                        title=f"{target_region} 연령별 매출 비중",
                        color_discrete_sequence=px.colors.sequential.Oranges
                    )
                     st.plotly_chart(fig_age_donut, use_container_width=True)
                
                with age_col2:
                    # 2. Dominant Age & Tactics
                    st.markdown(f"#### 🎯 핵심 타겟: {dominant_age}")
                    
                    tactics = {
                        "20대": "📱 **인스타/TikTok 숏폼 마케팅**: '감성 패키지'와 '가성비 못난이 과일' 소구 포인트 강조",
                        "30대": "🏢 **직장인/육아맘 타겟**: '아이 간식', '사무실 공동구매' 키워드로 맘카페 및 당근마켓 광고",
                        "40대": "👨‍👩‍👧‍👦 **가족 건강/선물**: '부모님 선물', '제철 보양' 메시지로 밴드(BAND) 및 카카오톡 선물하기 유도",
                        "50대": "🏔️ **동호회/커뮤니티**: 등산/골프 동호회 제휴 및 '단체 주문 할인' 프로모션 전개",
                        "60대 이상": "📞 **전화 주문/지인 추천**: 가독성 좋은 이미지 문자와 전화 주문 전용 핫라인 운영"
                    }
                    
                    selected_tactic = tactics.get(dominant_age, "모든 연령층을 아우르는 대중적인 마케팅 전개")
                    
                    st.info(f"**💡 {dominant_age} 맞춤 공략법**\n\n{selected_tactic}")
                    
                    # Show Top Product for this Age Group in this Region
                    age_specific_df = region_df[region_df['연령대'] == dominant_age]
                    if not age_specific_df.empty:
                        top_age_prod = age_specific_df.groupby('상품명')['실결제 금액'].sum().idxmax()
                        st.success(f"🔥 **{dominant_age} 최다 구매 상품**: {top_age_prod}")
            else:
                st.info("연령대 데이터가 없어 상세 전략을 수립할 수 없습니다.")

        else:
            st.warning(f"선택한 지역({target_region})의 데이터가 없습니다.")

    st.divider()
    st.subheader("🕰️ 주문 패턴 분석 (시간대/요일)")
    
    if not df_filtered.empty:
        # Preprocessing for Heatmap
        # Ensure '요일' is ordered correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Korean mapping if needed, but assuming data might have English or mixed. 
        # Let's check unique values or just use as is if already Korean. 
        # If '요일' is already Korean (월, 화...), day_order needs to match.
        # Check if Sample data uses Korean days based on previous view_file (line 117: dt.day_name() returns English by default unless locale set, but let's stick to observed data or handle gracefully)
        
        # Safe aggregation
        heatmap_data = df_filtered.groupby(['요일', '주문시간'])['주문수량'].sum().reset_index()
        
        # 1. Density Heatmap
        fig_heatmap = px.density_heatmap(
            heatmap_data, 
            x='주문시간', 
            y='요일', 
            z='주문수량', 
            nbinsx=24,
            category_orders={"요일": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] 
                             if heatmap_data['요일'].iloc[0] in ['Monday', 'Tuesday'] else 
                             ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]},
            color_continuous_scale='OrRd',
            title="요일 x 시간대별 주문 집중도 (Heatmap)"
        )
        fig_heatmap.update_layout(xaxis_title="시간대 (0~23시)", yaxis_title="요일")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 2. Peak Time Analysis
        heatmap_data['Hotscore'] = heatmap_data['주문수량']
        top_slots = heatmap_data.sort_values('Hotscore', ascending=False).head(3)
        
        if not top_slots.empty:
            st.markdown("#### ⚡ 골든 타임 (Golden Hours)")
            
            c1, c2, c3 = st.columns(3)
            for i, (idx, row) in enumerate(top_slots.iterrows()):
                with [c1, c2, c3][i]:
                    st.metric(
                        f"Top {i+1}", 
                        f"{row['요일']} {row['주문시간']}시", 
                        f"{row['주문수량']}건 주문"
                    )
            
            # 3. Actionable Advice
            best_day = top_slots.iloc[0]['요일']
            best_hour = top_slots.iloc[0]['주문시간']
            
            # Simple logic for advice
            target_hour = best_hour - 1 if best_hour > 0 else 23
            
            st.info(f"""
            **📢 마케팅 골든 타임 제안**
            
            가장 주문이 많은 **{best_day} {best_hour}시**를 공략하세요!
            - **PUSH 알림**: 1시간 전인 **{best_day} {target_hour}시**에 할인 쿠폰이나 타임 세일 알림을 보내면 전환율이 극대화될 수 있습니다.
            - **광고 입찰가 상향**: 이 시간대에 검색 광고 입찰가를 **20~30% 상향** 조정하여 노출을 늘리세요.
            """)
    else:
        st.warning("데이터가 없어 주문 패턴을 분석할 수 없습니다.")

elif "고객" in menu:
    # [View 4] 👥 고객 분석 Analysis (Comprehensive)
    st.header("👥 고객 데이터 분석 (Customer Intelligence)")

    if not df_filtered.empty:
        max_date = df_filtered['주문일'].max()
        
        # Tabs for organized view
        tab1, tab2, tab3 = st.tabs(["📊 기본 분석 (Basic)", "💎 고급 분석 (VIP/Retention)", "✨ 심층 인사이트 (Deep Dive)"])
        
        # --- Tab 1: Basic Analysis (Restored) ---
        with tab1:
            st.subheader("🔄 재구매율 및 인구통계 분석")
            
            # 1. Repurchase Rate
            if '재구매 횟수' in df_filtered.columns and 'UID' in df_filtered.columns:
                cust_stats = df_filtered.groupby('UID')['재구매 횟수'].max().reset_index()
                total_customers = len(cust_stats)
                returning_customers = len(cust_stats[cust_stats['재구매 횟수'] > 0])
                repurchase_rate = (returning_customers / total_customers * 100) if total_customers > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("전체 고객", f"{total_customers:,}명")
                c2.metric("재구매 고객", f"{returning_customers:,}명")
                c3.metric("재구매율", f"{repurchase_rate:.1f}%")
                
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    vals = [total_customers - returning_customers, returning_customers]
                    fig_pie = px.pie(
                        values=vals, 
                        names=['신규 (1회)', '재구매 (2회+)'], 
                        hole=0.4, 
                        title="신규 vs 재구매 비율",
                        color_discrete_sequence=['#E0E0E0', '#FF7F50']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col_chart2:
                    st.info("💡 **재구매 유도 팁**\n\n재구매율이 30% 미만이라면 '첫 구매 감사 쿠폰' 발송을 자동화해보세요.")

            st.divider()

            # 2. Age & Gender (if available)
            if '연령대' in df_filtered.columns:
                col_age1, col_age2 = st.columns(2)
                with col_age1:
                    age_sales = df_filtered.groupby('연령대')['실결제 금액'].sum().reset_index()
                    fig_age = px.pie(age_sales, values='실결제 금액', names='연령대', title="연령별 매출 비중", hole=0.4)
                    st.plotly_chart(fig_age, use_container_width=True)
                with col_age2:
                    age_aov = df_filtered.groupby('연령대')['실결제 금액'].mean().reset_index()
                    fig_aov = px.bar(age_aov, x='연령대', y='실결제 금액', title="연령별 객단가 비교", color='실결제 금액')
                    st.plotly_chart(fig_aov, use_container_width=True)
            else:
                st.warning("연령대 데이터가 없습니다.")

        # --- Tab 2: Advanced Analysis (RFM & Cohort) ---
        with tab2:
            # 1. RFM Segmentation
            st.subheader("💎 RFM 고객 세분화")
            
            rfm = df_filtered.groupby('UID').agg({
                '주문일': lambda x: (max_date - x.max()).days, # Recency
                '주문번호': 'count', # Frequency
                '실결제 금액': 'sum' # Monetary
            }).reset_index()
            rfm.rename(columns={'주문일': 'Recency', '주문번호': 'Frequency', '실결제 금액': 'Monetary'}, inplace=True)
            
            def assign_rfm_segment(row):
                if row['Recency'] > 90:
                    if row['Monetary'] > 200000: return '이탈 우려 (VIP)'
                    else: return '이탈 고객 (Lost)'
                else:
                    if row['Monetary'] > 300000: return 'VIP (최상위)'
                    elif row['Frequency'] >= 3: return '충성 고객 (Loyal)'
                    elif row['Recency'] <= 30: return '신규/최근 (New)'
                    else: return '일반 (Regular)'

            rfm['Segment'] = rfm.apply(assign_rfm_segment, axis=1)
            
            col_rfm1, col_rfm2 = st.columns([1, 1])
            with col_rfm1:
                seg_counts = rfm['Segment'].value_counts().reset_index()
                seg_counts.columns = ['Segment', 'Count']
                fig_rfm = px.pie(seg_counts, values='Count', names='Segment', title="고객 등급별 비중", hole=0.4)
                st.plotly_chart(fig_rfm, use_container_width=True)
            with col_rfm2:
                st.markdown("#### 📢 등급별 관리 전략")
                st.info("""
                - **💎 VIP**: 전용 핫라인 및 시크릿 쿠폰 제공
                - **💖 충성**: 정기 구독 서비스 제안
                - **🌱 신규**: 'n번째 구매' 달성 프로모션
                - **⚠️ 이탈**: 웰컴백 쿠폰 자동 발송
                """)

            st.divider()

            # 2. Cohort Analysis
            st.subheader("📅 코호트 잔존율 (Cohort Retention)")
            df_filtered['OrderMonth'] = df_filtered['주문일'].dt.to_period('M')
            df_filtered['CohortMonth'] = df_filtered.groupby('UID')['주문일'].transform('min').dt.to_period('M')
            
            cohort_data = df_filtered.groupby(['CohortMonth', 'OrderMonth'])['UID'].nunique().reset_index()
            cohort_data['Period'] = (cohort_data['OrderMonth'] - cohort_data['CohortMonth']).apply(lambda x: x.n)
            
            cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='Period', values='UID')
            cohort_size = cohort_pivot.iloc[:, 0]
            retention = cohort_pivot.divide(cohort_size, axis=0)
            
            fig_cohort = px.imshow(
                retention,
                labels=dict(x="경과 개월 수", y="가입 월", color="잔존율"),
                x=retention.columns,
                y=retention.index.astype(str),
                color_continuous_scale='Blues',
                text_auto='.1%'
            )
            st.plotly_chart(fig_cohort, use_container_width=True)

        # --- Tab 3: Additional Insights (New) ---
        with tab3:
            st.subheader("✨ 3가지 추가 인사이트 (Deep Dive)")
            
            # Insight 1: Geo-Distribution
            st.markdown("##### 1. 📍 지역별 고객 분포 (Top Regions)")
            if '지역' in df_filtered.columns:
                geo_dist = df_filtered.groupby('지역')['UID'].nunique().reset_index().sort_values('UID', ascending=False).head(10)
                fig_geo = px.bar(geo_dist, x='지역', y='UID', title="지역별 고객 수 Top 10", color='UID', color_continuous_scale='Viridis')
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.info("지역 데이터가 없어 분석할 수 없습니다.")
                
            st.divider()

            # Insight 2: Purchase Time Pattern (VIP vs Regular)
            st.markdown("##### 2. ⏰ VIP 고객의 주 구매 시간대")
            df_filtered['Hour'] = df_filtered['주문일'].dt.hour
            
            # Join segment info back to main df
            rfm_map = rfm[['UID', 'Segment']]
            df_seg = df_filtered.merge(rfm_map, on='UID', how='left')
            
            vip_hourly = df_seg[df_seg['Segment'].str.contains('VIP')].groupby('Hour')['주문번호'].count().reset_index()
            reg_hourly = df_seg[~df_seg['Segment'].str.contains('VIP')].groupby('Hour')['주문번호'].count().reset_index()
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=vip_hourly['Hour'], y=vip_hourly['주문번호'], mode='lines+markers', name='VIP 고객', line=dict(color='gold', width=3)))
            fig_time.add_trace(go.Scatter(x=reg_hourly['Hour'], y=reg_hourly['주문번호'], mode='lines', name='일반 고객', line=dict(color='grey', dash='dot')))
            fig_time.update_layout(title="시간대별 주문 패턴 비교 (VIP vs 일반)", xaxis_title="시간 (0~23시)", yaxis_title="주문 건수")
            st.plotly_chart(fig_time, use_container_width=True)
            st.caption("* VIP 고객이 활동하는 골든 타임을 파악하여 타임딜을 기획하세요.")
            
            st.divider()

            # Insight 3: Category Preference
            st.markdown("##### 3. 🛍️ VIP 선호 카테고리 (Category Preference)")
            # Assuming '카테고리' column exists, or use '상품명' top keywords if not
            target_col = '카테고리' if '카테고리' in df_filtered.columns else '상품명'
            
            vip_pref = df_seg[df_seg['Segment'].str.contains('VIP')][target_col].value_counts().head(5).reset_index()
            vip_pref.columns = [target_col, 'Count']
            
            fig_cat = px.bar(vip_pref, x='Count', y=target_col, orientation='h', title=f"VIP 고객이 가장 많이 산 {target_col}", color='Count')
            st.plotly_chart(fig_cat, use_container_width=True)

    else:
        st.warning("분석할 고객 데이터가 없습니다.")

elif "셀러" in menu:
    # [View 5] 📈 셀러 분석 Analysis (Advanced)
    st.header("📈 셀러 성과 및 관리 (Seller Management)")
    
    if not df_filtered.empty:
        # Pre-calc: Last Date in data
        max_date = df_filtered['주문일'].max()
        
        # 1. Seller Metrics Calculation
        seller_stats = df_filtered.groupby('셀러명').agg({
            '실결제 금액': 'sum',
            '주문수량': 'sum',
            '주문번호': 'count', # Order Count
            '주문일': 'max' # Last Active
        }).reset_index()
        
        seller_stats.rename(columns={'주문번호': '주문건수', '주문일': '최근활동일'}, inplace=True)
        seller_stats['객단가(AOV)'] = seller_stats['실결제 금액'] / seller_stats['주문건수']
        
        # 2. Seller Segmentation (S/A/B Grade)
        seller_stats = seller_stats.sort_values('실결제 금액', ascending=False)
        seller_stats['Cumulative Sales'] = seller_stats['실결제 금액'].cumsum()
        seller_stats['Cumulative Perc'] = seller_stats['Cumulative Sales'] / seller_stats['실결제 금액'].sum()
        
        def assign_seller_grade(row):
            if row['Cumulative Perc'] <= 0.10: return 'S (최상위)'
            elif row['Cumulative Perc'] <= 0.40: return 'A (우수)'
            else: return 'B (일반)'
            
        seller_stats['등급'] = seller_stats.apply(assign_seller_grade, axis=1)
        
        # 3. Growth Rate (Last 30 Days vs Previous 30 Days)
        t_current_start = max_date - timedelta(days=30)
        t_prev_start = t_current_start - timedelta(days=30)
        
        df_current = df_filtered[df_filtered['주문일'] >= t_current_start]
        df_prev = df_filtered[(df_filtered['주문일'] < t_current_start) & (df_filtered['주문일'] >= t_prev_start)]
        
        curr_sales = df_current.groupby('셀러명')['실결제 금액'].sum().reset_index().rename(columns={'실결제 금액': 'CurrentSales'})
        prev_sales = df_prev.groupby('셀러명')['실결제 금액'].sum().reset_index().rename(columns={'실결제 금액': 'PrevSales'})
        
        growth_df = curr_sales.merge(prev_sales, on='셀러명', how='outer').fillna(0)
        growth_df['GrowthRate'] = ((growth_df['CurrentSales'] - growth_df['PrevSales']) / growth_df['PrevSales'].replace(0, 1)) * 100
        
        # Merge Growth into Stats
        seller_stats = seller_stats.merge(growth_df[['셀러명', 'GrowthRate']], on='셀러명', how='left').fillna(0)

        # 4. Churn Risk (Dormant > 30 Days)
        seller_stats['DaysSinceActive'] = (max_date - seller_stats['최근활동일']).dt.days
        seller_stats['Status'] = seller_stats['DaysSinceActive'].apply(lambda x: '⚠️ 휴면 위험' if x >= 30 else '✅ 활동 중')
        
        # --- UI Rendering ---
        
        # Summary Metrics
        st.subheader("📊 셀러 현황 개요")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric("총 활동 셀러", f"{len(seller_stats)}명")
        with col_s2:
            s_grade_count = len(seller_stats[seller_stats['등급'].str.contains('S')])
            st.metric("S등급(상위 10%)", f"{s_grade_count}명")
        with col_s3:
            rising_stars = len(seller_stats[seller_stats['GrowthRate'] >= 20])
            st.metric("급성장 셀러 (MoM +20%↑)", f"{rising_stars}명")
        with col_s4:
            churn_risk = len(seller_stats[seller_stats['Status'].str.contains('위험')])
            st.metric("이탈 위험 (30일 무실적)", f"{churn_risk}명", delta=-churn_risk, delta_color="inverse")

        st.divider()

        col_main1, col_main2 = st.columns([1, 1])
        
        with col_main1:
            st.markdown("##### 🚀 라이징 스타 (Top Growth)")
            # Filter: Min 10 orders to avoid noise
            rising_df = seller_stats[seller_stats['주문건수'] >= 10].sort_values('GrowthRate', ascending=False).head(5)
            if not rising_df.empty:
                st.dataframe(
                    rising_df[['등급', '셀러명', 'GrowthRate', '실결제 금액']].style.format({
                        'GrowthRate': "{:+.1f}%", 
                        '실결제 금액': "{:,.0f}"
                    }),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("조건을 만족하는 성장 셀러가 없습니다.")
                
        with col_main2:
             st.markdown("##### ⚠️ 이탈 위험군 (Dormant)")
             dormant_df = seller_stats[seller_stats['Status'].str.contains('위험')].sort_values('DaysSinceActive', ascending=False).head(5)
             if not dormant_df.empty:
                 st.dataframe(
                    dormant_df[['등급', '셀러명', '최근활동일', 'DaysSinceActive']],
                    use_container_width=True, hide_index=True
                )
             else:
                 st.success("최근 30일 이내 활동하지 않은 셀러가 없습니다.")
        
        st.divider()

        # [Restored] 5. Monthly Seller Inflow & Churn Trend
        st.subheader("📆 월별 셀러 유입/이탈 추이")
        
        # Inflow Logic (First Order Date)
        first_dates = df_filtered.groupby('셀러명')['주문일'].min().reset_index()
        first_dates['Month'] = first_dates['주문일'].dt.to_period('M').astype(str)
        new_counts = first_dates.groupby('Month')['셀러명'].count().reset_index()
        
        fig_inflow = px.bar(
            new_counts, 
            x='Month', 
            y='셀러명', 
            title="월별 신규 셀러 유입 수 (New Sellers)", 
            labels={'셀러명': '신규 셀러 수', 'Month': '월'},
            color_discrete_sequence=['#2ECC71']
        )
        st.plotly_chart(fig_inflow, use_container_width=True)
        
        # 6. Strategic Insights (Sales & Acquisition)
        st.divider()
        st.subheader("💡 셀러 성장 및 확보 전략 (Strategy Report)")
        
        strat_col1, strat_col2 = st.columns(2)
        
        with strat_col1:
            st.markdown("#### 🌱 등급별 판매 증대 가이드")
            st.info("""
            **👑 S등급 (상위 10% - 선도 그룹)**
            - **전략**: `브랜드 팬덤 구축` 및 `객단가(AOV) 극대화`
            - **액션**: 프리미엄 라인업 단독 기획전, VIP 전용 '선물하기' 패키지 개발지원
            
            **🚀 A등급 (상위 30% - 성장 그룹)**
            - **전략**: `구매 전환율 개선` 및 `재구매 유도`
            - **액션**: 베스트 상품 리뷰 이벤트, '첫 구매 후 1개월 내 재구매' 쿠폰 발송 자동화
            
            **🌱 B등급 (일반 - 육성 그룹)**
            - **전략**: `상품 노출 확대` 및 `기초 세팅 최적화`
            - **액션**: 썸네일/상세페이지 무료 진단 컨설팅, 검색광고(CPC) 소액 지원 프로모션
            """)
            
        with strat_col2:
            st.markdown("#### 📢 신규 셀러 확보(Acquisition) 전술")
            
            # Analyze recent inflow trend
            recent_months = new_counts.sort_values('Month', ascending=False).head(2)
            if len(recent_months) >= 2:
                last_month_cnt = recent_months.iloc[0]['셀러명']
                prev_month_cnt = recent_months.iloc[1]['셀러명']
                
                if last_month_cnt < prev_month_cnt:
                   status_msg = f"📉 **경고**: 지난달 대비 신규 유입이 감소했습니다. ({prev_month_cnt}명 → {last_month_cnt}명)"
                   action_msg = """
                   - **추천인 보상 강화**: 기존 셀러가 신규 셀러 추천 시 '판매 수수료 1개월 면제' 혜택 제공
                   - **파워블로거/유튜버 제휴**: '농산물 판매 성공 사례' 콘텐츠 제작 및 배포
                   """
                   st.warning(f"{status_msg}\n{action_msg}")
                else:
                   status_msg = f"📈 **양호**: 신규 유입이 증가하거나 유지되고 있습니다. ({prev_month_cnt}명 → {last_month_cnt}명)"
                   action_msg = """
                   - **온보딩 프로세스 최적화**: 가입 후 첫 상품 등록까지 걸리는 시간을 단축시키세요.
                   - **초기 정착 지원**: '신규 입점 웰컴 키트' (포장재 샘플 등) 제공으로 이탈 방지
                   """
                   st.success(f"{status_msg}\n{action_msg}")
            else:
                st.info("데이터가 부족하여 유입 추이를 분석할 수 없습니다.")
        
        st.divider()

        # 7. Market Basket Analysis (Bundle Strategies) [NEW/REPLACEMENT]
        st.subheader("🛒 장바구니 분석 (Market Basket Analysis)")
        st.markdown("고객의 **동시 구매 패턴**을 분석하여 객단가(AOV)를 높일 수 있는 **꿀조합 상품**을 제안합니다.")
        
        if '주문번호' in df_filtered.columns and '상품명' in df_filtered.columns:
            # 7-1. Single vs Multi-item Order Analysis
            order_counts = df_filtered.groupby('주문번호')['상품명'].count()
            multi_item_orders = order_counts[order_counts > 1].index
            single_item_orders = order_counts[order_counts == 1].index
            
            multi_aov = df_filtered[df_filtered['주문번호'].isin(multi_item_orders)]['실결제 금액'].sum() / len(multi_item_orders) if len(multi_item_orders) > 0 else 0
            single_aov = df_filtered[df_filtered['주문번호'].isin(single_item_orders)]['실결제 금액'].sum() / len(single_item_orders) if len(single_item_orders) > 0 else 0
            
            c_b1, c_b2, c_b3 = st.columns(3)
            c_b1.metric("단품 주문 비중", f"{(len(single_item_orders)/len(order_counts)*100):.1f}%")
            c_b2.metric("합배송(세트) 주문 비중", f"{(len(multi_item_orders)/len(order_counts)*100):.1f}%")
            c_b3.metric("세트 구매시 객단가 효과", f"+{((multi_aov - single_aov)/single_aov*100):.1f}%", delta_color="normal")
            
            st.info(f"💡 고객이 상품을 묶어 살 때, 단품 구매보다 객단가가 약 **{int(multi_aov - single_aov):,}원** 더 높습니다. 세트 상품 구성이 필수적입니다.")
            
            # 7-2. Top Synergy Pairs (Co-occurrence)
            from itertools import combinations
            from collections import Counter
            
            # Get list of products per order (only for multi-item orders)
            # Optimization: Limit to top 1000 orders if too slow, but dataset seems small enough considering context
            multi_order_df = df_filtered[df_filtered['주문번호'].isin(multi_item_orders)]
            
            # Group items by order
            basket_lists = multi_order_df.groupby('주문번호')['상품명'].apply(list)
            
            pair_counter = Counter()
            for items in basket_lists:
                items = sorted(items) # Sort to ensure (A, B) is same as (B, A)
                pair_counter.update(combinations(items, 2))
                
            top_pairs = pair_counter.most_common(5)
            
            st.markdown("##### 🤝 함께 사면 좋은 '꿀조합' Top 5 (Synergy Pairs)")
            
            if top_pairs:
                pair_data = []
                for (item1, item2), count in top_pairs:
                    pair_data.append({
                        '상품 A': item1,
                        '상품 B': item2,
                        '동시 구매 횟수': count,
                        '추천 전략': '번들 할인 패키지 구성 (5~10% 할인)'
                    })
                st.dataframe(pd.DataFrame(pair_data), use_container_width=True, hide_index=True)
            else:
                st.warning("동시 구매 데이터가 충분하지 않아 조합을 추천할 수 없습니다.")
                
        else:
            st.warning("주문번호 또는 상품명 데이터가 없어 장바구니 분석을 수행할 수 없습니다.")

        st.divider()
        
        # Detailed Table
        st.markdown("##### 📋 전체 셀러 상세 지표")
        
        # Clean col names for display
        display_cols = ['등급', '셀러명', '실결제 금액', 'GrowthRate', '주문건수', '객단가(AOV)', '최근활동일', 'Status']
        
        st.dataframe(
            seller_stats[display_cols].style.format({
                '실결제 금액': "{:,.0f}",
                'GrowthRate': "{:+.1f}%",
                '주문건수': "{:,.0f}",
                '객단가(AOV)': "{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
    else:
        st.warning("분석할 셀러 데이터가 없습니다.")
