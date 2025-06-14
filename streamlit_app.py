import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import boto3
from cachetools import cached, TTLCache
import requests
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import os
from typing import Dict, List, Tuple, Any

# Cache for pricing data
cache = TTLCache(maxsize=100, ttl=3600)

# Initialize Anthropic client
@st.cache_resource
def get_claude_client():
    """Initialize Claude AI client"""
    api_key = os.getenv('ANTHROPIC_API_KEY') or st.secrets.get('ANTHROPIC_API_KEY')
    if not api_key:
        st.warning("⚠️ Anthropic API key not found. Please set ANTHROPIC_API_KEY in your environment or Streamlit secrets.")
        return None
    return anthropic.Anthropic(api_key=api_key)

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Oracle to MongoDB Migration Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

# Custom CSS (enhanced for AI features)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ai-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9900;
    }
    .cost-savings {
        color: #28a745;
        font-weight: bold;
    }
    .cost-increase {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-assessment {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .migration-step {
        background-color: #e8f4f8;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #17a2b8;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Claude AI Integration Functions
def generate_ai_migration_analysis(servers: Dict, data_size_tb: float, complexity_score: int, 
                                 num_pl_sql_objects: int, cost_analysis: pd.DataFrame) -> str:
    """Generate comprehensive migration analysis using Claude AI"""
    client = get_claude_client()
    if not client:
        return "AI analysis unavailable - API key not configured."
    
    prompt = f"""
    Analyze this Oracle to MongoDB migration scenario and provide detailed insights:

    **Infrastructure Overview:**
    - Environments: {list(servers.keys())}
    - Total Data Size: {data_size_tb} TB
    - Complexity Score: {complexity_score}/100
    - PL/SQL Objects: {num_pl_sql_objects}
    
    **Environment Details:**
    {json.dumps(servers, indent=2)}
    
    **Cost Analysis Summary:**
    - Total Current Annual Cost: ${cost_analysis['Current_Total'].sum():,.0f}
    - Total Projected AWS Cost: ${cost_analysis['AWS_Total_Cost'].sum():,.0f}
    - Total Annual Savings: ${cost_analysis['Annual_Savings'].sum():,.0f}
    
    Please provide:
    1. **Migration Feasibility Assessment** - Rate the migration complexity and provide reasoning
    2. **Key Technical Challenges** - Identify the top 5 technical hurdles
    3. **Risk Mitigation Strategies** - Specific strategies for each identified risk
    4. **Performance Optimization Recommendations** - MongoDB-specific optimizations
    5. **Cost Optimization Opportunities** - Additional ways to reduce costs
    6. **Timeline Recommendations** - Realistic timeline with milestones
    7. **Success Metrics** - KPIs to track migration success
    
    Format your response in clear sections with actionable recommendations.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

def generate_plsql_conversion_strategy(num_pl_sql_objects: int, complexity_score: int) -> str:
    """Generate PL/SQL conversion strategy using Claude AI"""
    client = get_claude_client()
    if not client:
        return "AI analysis unavailable - API key not configured."
    
    prompt = f"""
    Create a detailed PL/SQL to MongoDB conversion strategy for:
    - Number of PL/SQL Objects: {num_pl_sql_objects}
    - Overall Complexity Score: {complexity_score}/100
    
    Provide:
    1. **Conversion Approach** - Best practices for PL/SQL migration
    2. **Common PL/SQL Patterns** - How to handle procedures, functions, triggers
    3. **MongoDB Alternatives** - Aggregation pipelines, server-side JavaScript
    4. **Code Refactoring Strategy** - Step-by-step approach
    5. **Testing Strategy** - How to validate converted logic
    6. **Performance Considerations** - Optimization techniques
    7. **Tools and Resources** - Recommended migration tools
    
    Be specific and provide code examples where helpful.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating PL/SQL strategy: {str(e)}"

def generate_security_compliance_analysis(servers: Dict, high_availability: bool) -> str:
    """Generate security and compliance analysis using Claude AI"""
    client = get_claude_client()
    if not client:
        return "AI analysis unavailable - API key not configured."
    
    prompt = f"""
    Analyze security and compliance requirements for Oracle to MongoDB migration:
    
    **Environment Setup:**
    - Environments: {list(servers.keys())}
    - High Availability Required: {high_availability}
    
    **Production Environment:**
    {json.dumps(servers.get('Prod', servers.get('Production', {})), indent=2)}
    
    Provide analysis on:
    1. **Security Architecture** - AWS security best practices for MongoDB
    2. **Compliance Requirements** - GDPR, HIPAA, SOX considerations
    3. **Data Encryption** - At-rest and in-transit encryption strategies
    4. **Access Control** - IAM, VPC, and MongoDB security
    5. **Monitoring & Auditing** - Security monitoring setup
    6. **Backup & Recovery** - Secure backup strategies
    7. **Network Security** - VPC design and security groups
    
    Focus on production-ready security configurations.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating security analysis: {str(e)}"

def generate_performance_optimization_plan(servers: Dict, data_size_tb: float) -> str:
    """Generate performance optimization plan using Claude AI"""
    client = get_claude_client()
    if not client:
        return "AI analysis unavailable - API key not configured."
    
    prod_specs = servers.get('Prod', servers.get('Production', list(servers.values())[-1]))
    
    prompt = f"""
    Create a performance optimization plan for MongoDB migration:
    
    **Production Environment:**
    - CPU Cores: {prod_specs.get('cpu', 'N/A')}
    - RAM: {prod_specs.get('ram', 'N/A')} GB
    - Storage: {prod_specs.get('storage', 'N/A')} GB
    - IOPS: {prod_specs.get('throughput', 'N/A')}
    - Data Size: {data_size_tb} TB
    
    Provide recommendations for:
    1. **Schema Design Optimization** - Document structure best practices
    2. **Indexing Strategy** - Optimal index design for performance
    3. **Sharding Strategy** - Horizontal scaling approach
    4. **Connection Pooling** - Application-level optimizations
    5. **Query Optimization** - MongoDB query patterns
    6. **Memory Optimization** - Working set and cache management
    7. **I/O Optimization** - Storage and throughput optimization
    8. **Monitoring Setup** - Performance monitoring tools and metrics
    
    Include specific MongoDB configuration recommendations.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating performance plan: {str(e)}"

def generate_detailed_migration_roadmap(servers: Dict, timeline_months: int, complexity_score: int) -> str:
    """Generate detailed migration roadmap using Claude AI"""
    client = get_claude_client()
    if not client:
        return "AI analysis unavailable - API key not configured."
    
    prompt = f"""
    Create a detailed migration roadmap for Oracle to MongoDB:
    
    **Project Parameters:**
    - Timeline: {timeline_months} months
    - Complexity Score: {complexity_score}/100
    - Environments: {list(servers.keys())}
    
    Create a week-by-week breakdown including:
    1. **Phase 1: Discovery & Planning** (Weeks 1-4)
    2. **Phase 2: Architecture & Design** (Weeks 5-8)
    3. **Phase 3: Development & Testing** (Weeks 9-16)
    4. **Phase 4: Migration Execution** (Weeks 17-20)
    5. **Phase 5: Go-Live & Optimization** (Weeks 21-24)
    
    For each phase, include:
    - Key deliverables
    - Team responsibilities
    - Risk mitigation activities
    - Success criteria
    - Dependencies and blockers
    
    Provide a realistic timeline with buffer for unexpected issues.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating migration roadmap: {str(e)}"

# Enhanced existing functions with AI insights
def analyze_complexity_with_ai(servers, data_size_tb, num_pl_sql_objects=0, num_applications=1):
    """Enhanced complexity analysis with AI insights"""
    # Original complexity calculation
    prod_specs = servers.get('Prod', servers[list(servers.keys())[-1]])
    
    cpu_complexity = min(100, prod_specs['cpu'] * 3)
    ram_complexity = min(100, prod_specs['ram'] / 4)
    storage_complexity = min(100, prod_specs['storage'] / 100)
    throughput_complexity = min(100, prod_specs['throughput'] / 1000)
    data_complexity = min(100, data_size_tb * 3)
    env_complexity = min(100, len(servers) * 15)
    plsql_complexity = min(100, num_pl_sql_objects / 10) if num_pl_sql_objects > 0 else 30
    app_complexity = min(100, num_applications * 20)
    
    weights = {
        'cpu': 0.15, 'ram': 0.15, 'storage': 0.10, 'throughput': 0.15,
        'data_size': 0.15, 'environments': 0.10, 'plsql': 0.15, 'applications': 0.05
    }
    
    total_score = sum(locals()[f"{key}_complexity"] * weight for key, weight in weights.items())
    
    details = {
        'CPU Cores': round(cpu_complexity),
        'RAM Size': round(ram_complexity),
        'Storage Size': round(storage_complexity),
        'Throughput Requirements': round(throughput_complexity),
        'Data Volume': round(data_complexity),
        'Multiple Environments': round(env_complexity),
        'PL/SQL Objects': round(plsql_complexity),
        'Application Count': round(app_complexity)
    }
    
    # AI-enhanced risk assessment
    risk_factors = []
    if plsql_complexity > 60:
        risk_factors.append("High PL/SQL complexity requires significant code refactoring")
    if data_complexity > 70:
        risk_factors.append("Large data volume increases migration time and complexity")
    if env_complexity > 50:
        risk_factors.append("Multiple environments require coordinated migration strategy")
    if throughput_complexity > 80:
        risk_factors.append("High throughput requirements need careful performance tuning")
    
    return min(100, round(total_score)), details, risk_factors

# [Previous functions remain the same - get_ec2_prices, get_ebs_prices, etc.]

# Main Streamlit App with AI Enhancement
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Powered Oracle to MongoDB Migration Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Anthropic Claude AI for Intelligent Migration Analysis")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AI Features Toggle
        st.subheader("🤖 AI Analysis Features")
        enable_ai_analysis = st.checkbox("Enable AI-Powered Analysis", True)
        ai_analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Basic", "Comprehensive", "Expert"],
            index=1
        )
        
        # AWS Region Selection
        aws_regions = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)'
        }
        selected_region = st.selectbox("Select AWS Region", list(aws_regions.keys()), 
                                     format_func=lambda x: aws_regions[x])
        
        st.markdown("---")
        
        # [Rest of the sidebar configuration remains the same...]
        
        with st.form("migration_config"):
            st.subheader("🏢 Environment Specifications")
            
            # Number of environments
            num_environments = st.number_input("Number of Environments", 1, 10, 5)
            
            # Environment details
            servers = {}
            default_envs = ['Dev', 'QA', 'UAT', 'PreProd', 'Prod']
            
            for i in range(num_environments):
                env_name = st.text_input(f"Environment {i+1} Name", 
                                       default_envs[i] if i < len(default_envs) else f"Env{i+1}")
                
                col1, col2 = st.columns(2)
                with col1:
                    cpu = st.number_input(f"CPU Cores ({env_name})", 1, 128, 
                                        [2, 4, 8, 16, 32][min(i, 4)])
                    ram = st.number_input(f"RAM (GB) ({env_name})", 4, 1024, 
                                        [8, 16, 32, 64, 128][min(i, 4)])
                
                with col2:
                    storage = st.number_input(f"Storage (GB) ({env_name})", 10, 10000, 
                                            [100, 200, 500, 1000, 2000][min(i, 4)])
                    throughput = st.number_input(f"Throughput (IOPS) ({env_name})", 100, 100000, 
                                               [1000, 2000, 5000, 10000, 20000][min(i, 4)])
                
                daily_usage = st.slider(f"Daily Usage Hours ({env_name})", 1, 24, 
                                      [8, 12, 16, 20, 24][min(i, 4)])
                
                servers[env_name] = {
                    'cpu': cpu, 'ram': ram, 'storage': storage,
                    'throughput': throughput, 'daily_usage': daily_usage
                }
            
            st.markdown("---")
            st.subheader("💰 Cost Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                oracle_license_cost = st.number_input("Oracle License Cost ($/year)", 0, 2000000, 100000)
                manpower_cost = st.number_input("Maintenance Cost ($/year)", 0, 1000000, 150000)
            
            with col2:
                data_size_tb = st.number_input("Total Data Size (TB)", 1, 1000, 25)
                migration_timeline = st.slider("Migration Timeline (Months)", 1, 18, 6)
            
            st.markdown("---")
            st.subheader("🔧 Technical Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                num_pl_sql_objects = st.number_input("Number of PL/SQL Objects", 0, 10000, 500)
                num_applications = st.number_input("Number of Applications", 1, 50, 3)
            
            with col2:
                backup_retention = st.selectbox("Backup Retention (Days)", [7, 14, 30, 90], index=2)
                high_availability = st.checkbox("High Availability Required", True)
            
            analyze_button = st.form_submit_button("🔍 Analyze Migration with AI", type="primary")
    
    # Main Content
    if analyze_button:
        # Show loading with AI analysis
        with st.spinner('🤖 Performing AI-powered migration analysis...'):
            # [Previous analysis code remains the same...]
            
            # Fetch pricing data
            ec2_prices = get_ec2_prices(selected_region)
            ebs_prices = get_ebs_prices(selected_region)
            rds_prices = get_rds_prices(selected_region)
            mongodb_prices = get_mongodb_atlas_prices()
            
            # Generate recommendations
            recommendations = recommend_instances(servers)
            
            # Calculate costs
            cost_df = calculate_comprehensive_costs(
                servers, ec2_prices, ebs_prices, rds_prices, mongodb_prices,
                oracle_license_cost, manpower_cost, data_size_tb, recommendations
            )
            
            # Enhanced complexity analysis with AI
            complexity_score, complexity_details, risk_factors = analyze_complexity_with_ai(
                servers, data_size_tb, num_pl_sql_objects, num_applications
            )
            
            # Generate AI analyses
            ai_analyses = {}
            if enable_ai_analysis:
                with st.spinner('🧠 Generating AI insights...'):
                    ai_analyses = {
                        'migration_analysis': generate_ai_migration_analysis(
                            servers, data_size_tb, complexity_score, num_pl_sql_objects, cost_df
                        ),
                        'plsql_strategy': generate_plsql_conversion_strategy(
                            num_pl_sql_objects, complexity_score
                        ),
                        'security_analysis': generate_security_compliance_analysis(
                            servers, high_availability
                        ),
                        'performance_plan': generate_performance_optimization_plan(
                            servers, data_size_tb
                        ),
                        'migration_roadmap': generate_detailed_migration_roadmap(
                            servers, migration_timeline, complexity_score
                        )
                    }
        
        # Display Results with AI Enhancement
        st.success("✅ AI-Powered Analysis Complete!")
        
        # Executive Summary
        st.header("📊 Executive Summary")
        
        total_current_cost = cost_df['Current_Total'].sum()
        total_aws_cost = cost_df['AWS_Total_Cost'].sum()
        total_annual_savings = cost_df['Annual_Savings'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Annual Cost", f"${total_current_cost:,.0f}")
        with col2:
            st.metric("Projected AWS Cost", f"${total_aws_cost:,.0f}")
        with col3:
            st.metric("Annual Savings", f"${total_annual_savings:,.0f}", 
                     delta=f"{(total_annual_savings/total_current_cost*100):.1f}%")
        with col4:
            st.metric("Complexity Score", f"{complexity_score}/100", 
                     delta="High" if complexity_score > 70 else "Medium" if complexity_score > 40 else "Low")
        
        # AI Migration Analysis
        if enable_ai_analysis and 'migration_analysis' in ai_analyses:
            st.markdown("---")
            st.header("🤖 AI Migration Analysis")
            st.markdown('<div class="ai-insight">', unsafe_allow_html=True)
            st.markdown("### Claude AI Insights")
            st.markdown(ai_analyses['migration_analysis'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk Assessment with AI
        if risk_factors:
            st.markdown("---")
            st.header("⚠️ Risk Assessment")
            st.markdown('<div class="risk-assessment">', unsafe_allow_html=True)
            st.markdown("### Identified Risk Factors:")
            for risk in risk_factors:
                st.markdown(f"• {risk}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed AI Analyses Tabs
        if enable_ai_analysis:
            st.markdown("---")
            st.header("🔍 Detailed AI Analysis")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "PL/SQL Strategy", "Security & Compliance", 
                "Performance Optimization", "Migration Roadmap", "Cost Analysis"
            ])
            
            with tab1:
                st.markdown("### PL/SQL Conversion Strategy")
                if 'plsql_strategy' in ai_analyses:
                    st.markdown(ai_analyses['plsql_strategy'])
                else:
                    st.info("Enable AI analysis to get PL/SQL conversion strategy")
            
            with tab2:
                st.markdown("### Security & Compliance Analysis")
                if 'security_analysis' in ai_analyses:
                    st.markdown(ai_analyses['security_analysis'])
                else:
                    st.info("Enable AI analysis to get security recommendations")
            
            with tab3:
                st.markdown("### Performance Optimization Plan")
                if 'performance_plan' in ai_analyses:
                    st.markdown(ai_analyses['performance_plan'])
                else:
                    st.info("Enable AI analysis to get performance optimization plan")
            
            with tab4:
                st.markdown("### Detailed Migration Roadmap")
                if 'migration_roadmap' in ai_analyses:
                    st.markdown(ai_analyses['migration_roadmap'])
                else:
                    st.info("Enable AI analysis to get detailed migration roadmap")
            
            with tab5:
                st.markdown("### Cost Analysis Details")
                st.dataframe(cost_df, use_container_width=True)
        
        # [Previous visualization and analysis code remains the same...]
        
        # Generate PDF Report with AI Insights
        if st.button("📄 Generate AI-Enhanced Migration Report"):
            with st.spinner("Generating comprehensive migration report..."):
                # Implementation for PDF report generation with AI insights
                st.success("Report generated successfully!")
                st.download_button(
                    label="Download Migration Report",
                    data="report_content",  # Implement actual PDF generation
                    file_name=f"migration_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    # Setup instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Setup Instructions")
    st.sidebar.markdown("""
    **To enable AI features:**
    
    1. **Get Anthropic API Key:**
       - Visit [Anthropic Console](https://console.anthropic.com/)
       - Create account and get API key
    
    2. **Set Environment Variable:**
       ```bash
       export ANTHROPIC_API_KEY=your_api_key_here
       ```
    
    3. **Or use Streamlit Secrets:**
       - Add to `.streamlit/secrets.toml`:
       ```toml
       ANTHROPIC_API_KEY = "your_api_key_here"
       ```
    
    4. **Install Required Packages:**
       ```bash
       pip install anthropic streamlit pandas plotly
       ```
    """)
    
    main()