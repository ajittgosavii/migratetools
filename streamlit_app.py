import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import requests
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Remove problematic imports and use fallback implementations
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    st.warning("Anthropic package not available. Install with: pip install anthropic")

try:
    import boto3
    AWS_API_AVAILABLE = True
except ImportError:
    AWS_API_AVAILABLE = False
    st.info("AWS SDK not available. Using fallback pricing data.")

# Page Configuration
st.set_page_config(
    page_title="Oracle to MongoDB Migration Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Custom CSS
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
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fallback pricing data (current as of 2024)
@st.cache_data
def get_fallback_pricing_data():
    """Get fallback pricing data when AWS API is not available"""
    return {
        'ec2': {
            't3.medium': 0.0416,
            't3.large': 0.0832,
            't3.xlarge': 0.1664,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'r5.large': 0.126,
            'r5.xlarge': 0.252,
            'r5.2xlarge': 0.504,
            'r5.4xlarge': 1.008,
            'r5.8xlarge': 2.016
        },
        'ebs': {
            'gp3': 0.08,   # per GB-month
            'gp2': 0.10,   # per GB-month
            'io1': 0.125,  # per GB-month
            'io2': 0.125,  # per GB-month
        },
        'rds': {
            'db.t3.medium': 0.068,
            'db.t3.large': 0.136,
            'db.m5.large': 0.192,
            'db.m5.xlarge': 0.384,
            'db.m5.2xlarge': 0.768,
            'db.r5.large': 0.240,
            'db.r5.xlarge': 0.480,
            'db.r5.2xlarge': 0.960,
            'db.r5.4xlarge': 1.920
        },
        'mongodb_atlas': {
            'M10': 0.08,   # per hour
            'M20': 0.12,   # per hour
            'M30': 0.54,   # per hour
            'M40': 1.08,   # per hour
            'M50': 2.16,   # per hour
            'M60': 4.32,   # per hour
            'M80': 8.64,   # per hour
            'M140': 17.28, # per hour
            'M200': 25.92, # per hour
            'M300': 43.20  # per hour
        }
    }

# AI Integration (with fallback)
def get_claude_analysis(prompt: str, analysis_type: str = "general") -> str:
    """Get Claude AI analysis with fallback to rule-based analysis"""
    if not ANTHROPIC_AVAILABLE:
        return get_fallback_analysis(analysis_type)
    
    try:
        api_key = st.secrets.get('ANTHROPIC_API_KEY') or st.session_state.get('anthropic_api_key')
        if not api_key:
            return get_fallback_analysis(analysis_type)
        
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.warning(f"AI analysis unavailable: {str(e)}")
        return get_fallback_analysis(analysis_type)

def get_fallback_analysis(analysis_type: str) -> str:
    """Fallback analysis when AI is not available"""
    fallback_analyses = {
        "migration": """
        **Migration Feasibility Assessment**
        Based on your configuration, this migration appears feasible with moderate complexity.
        
        **Key Technical Challenges:**
        1. PL/SQL code conversion to application logic
        2. Schema design optimization for MongoDB
        3. Data migration and validation
        4. Performance tuning and optimization
        5. Application integration updates
        
        **Recommended Approach:**
        - Start with non-production environments
        - Implement incremental migration strategy
        - Focus on thorough testing at each stage
        - Plan for parallel running during transition
        
        **Timeline Considerations:**
        Allow 20-30% buffer time for unexpected challenges and thorough testing.
        """,
        
        "plsql": """
        **PL/SQL Conversion Strategy**
        
        **Conversion Approach:**
        1. **Stored Procedures** → Application business logic
        2. **Functions** → Utility functions in application layer
        3. **Triggers** → Application event handlers or MongoDB change streams
        4. **Packages** → Service layer components
        
        **MongoDB Alternatives:**
        - Use aggregation pipelines for complex data processing
        - Implement business logic in application layer
        - Use MongoDB transactions for ACID requirements
        - Consider server-side JavaScript for simple operations
        
        **Testing Strategy:**
        - Unit test all converted logic
        - Create data validation scripts
        - Performance test critical operations
        - Implement comprehensive regression testing
        """,
        
        "security": """
        **Security & Compliance Analysis**
        
        **Security Architecture:**
        1. **Network Security:** VPC with private subnets, security groups
        2. **Access Control:** IAM roles, MongoDB authentication
        3. **Encryption:** TLS in transit, encryption at rest
        4. **Monitoring:** CloudWatch, MongoDB Compass monitoring
        
        **Compliance Considerations:**
        - Implement data encryption for sensitive information
        - Set up audit logging for compliance requirements
        - Configure backup and retention policies
        - Document security procedures and access controls
        
        **Best Practices:**
        - Use least privilege access principles
        - Regular security assessments and updates
        - Implement monitoring and alerting
        - Plan for disaster recovery scenarios
        """,
        
        "performance": """
        **Performance Optimization Plan**
        
        **Schema Design:**
        1. Design documents to minimize joins
        2. Embed related data when appropriate
        3. Use proper data types and structures
        4. Plan for future scalability needs
        
        **Indexing Strategy:**
        - Create compound indexes for query patterns
        - Use partial indexes for selective queries
        - Monitor index usage and performance
        - Regular index maintenance and optimization
        
        **Query Optimization:**
        - Use aggregation pipelines efficiently
        - Implement proper query patterns
        - Monitor slow queries and optimize
        - Use explain plans for performance analysis
        
        **Scaling Considerations:**
        - Plan sharding strategy for large datasets
        - Configure replica sets for high availability
        - Implement connection pooling
        - Monitor resource utilization
        """
    }
    
    return fallback_analyses.get(analysis_type, "Analysis not available in fallback mode.")

def analyze_complexity(servers, data_size_tb, num_pl_sql_objects=0, num_applications=1):
    """Analyze migration complexity"""
    prod_specs = servers.get('Prod', servers.get('Production', list(servers.values())[-1]))
    
    # Calculate complexity factors (0-100 scale)
    cpu_complexity = min(100, prod_specs['cpu'] * 3)
    ram_complexity = min(100, prod_specs['ram'] / 4)
    storage_complexity = min(100, prod_specs['storage'] / 100)
    throughput_complexity = min(100, prod_specs['throughput'] / 1000)
    data_complexity = min(100, data_size_tb * 3)
    env_complexity = min(100, len(servers) * 15)
    plsql_complexity = min(100, num_pl_sql_objects / 10) if num_pl_sql_objects > 0 else 30
    app_complexity = min(100, num_applications * 20)
    
    # Weighted average
    weights = {
        'cpu': 0.15, 'ram': 0.15, 'storage': 0.10, 'throughput': 0.15,
        'data_size': 0.15, 'environments': 0.10, 'plsql': 0.15, 'applications': 0.05
    }
    
    total_score = (
        cpu_complexity * weights['cpu'] +
        ram_complexity * weights['ram'] +
        storage_complexity * weights['storage'] +
        throughput_complexity * weights['throughput'] +
        data_complexity * weights['data_size'] +
        env_complexity * weights['environments'] +
        plsql_complexity * weights['plsql'] +
        app_complexity * weights['applications']
    )
    
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
    
    # Risk factors
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

def recommend_instances(servers):
    """Recommend AWS instances and MongoDB clusters"""
    recommendations = {}
    
    for env, specs in servers.items():
        cpu = specs['cpu']
        ram = specs['ram']
        
        # EC2 recommendations based on CPU and RAM requirements
        if env.lower() in ['dev', 'development']:
            if cpu <= 2 and ram <= 8:
                recommendations[env] = {'ec2': 't3.medium', 'mongodb': 'M10'}
            elif cpu <= 4 and ram <= 16:
                recommendations[env] = {'ec2': 't3.large', 'mongodb': 'M20'}
            else:
                recommendations[env] = {'ec2': 'm5.large', 'mongodb': 'M30'}
        
        elif env.lower() in ['qa', 'test', 'testing']:
            if cpu <= 4 and ram <= 16:
                recommendations[env] = {'ec2': 't3.large', 'mongodb': 'M20'}
            elif cpu <= 8 and ram <= 32:
                recommendations[env] = {'ec2': 'm5.xlarge', 'mongodb': 'M30'}
            else:
                recommendations[env] = {'ec2': 'm5.2xlarge', 'mongodb': 'M40'}
        
        elif env.lower() in ['uat', 'staging']:
            if cpu <= 8 and ram <= 32:
                recommendations[env] = {'ec2': 'r5.xlarge', 'mongodb': 'M40'}
            elif cpu <= 16 and ram <= 64:
                recommendations[env] = {'ec2': 'r5.2xlarge', 'mongodb': 'M50'}
            else:
                recommendations[env] = {'ec2': 'r5.4xlarge', 'mongodb': 'M60'}
        
        else:  # Production and PreProd
            if cpu <= 16 and ram <= 64:
                recommendations[env] = {'ec2': 'r5.2xlarge', 'mongodb': 'M50'}
            elif cpu <= 32 and ram <= 128:
                recommendations[env] = {'ec2': 'r5.4xlarge', 'mongodb': 'M60'}
            else:
                recommendations[env] = {'ec2': 'r5.8xlarge', 'mongodb': 'M80'}
    
    return recommendations

def calculate_costs(servers, pricing_data, oracle_license, manpower, data_size_tb, recommendations):
    """Calculate comprehensive cost analysis"""
    cost_data = []
    total_envs = len(servers)
    
    for env, specs in servers.items():
        # Current Oracle costs (distributed across environments)
        oracle_license_per_env = oracle_license / total_envs
        manpower_per_env = manpower / total_envs
        
        # Oracle RDS costs (estimate)
        oracle_rds_instance = 'db.r5.2xlarge'
        if specs['cpu'] <= 8:
            oracle_rds_instance = 'db.m5.xlarge'
        elif specs['cpu'] <= 16:
            oracle_rds_instance = 'db.r5.xlarge'
        
        oracle_rds_cost = pricing_data['rds'].get(oracle_rds_instance, 0.960) * specs['daily_usage'] * 365
        oracle_storage_cost = pricing_data['ebs']['gp2'] * specs['storage'] * 12
        
        current_total_cost = oracle_license_per_env + manpower_per_env + oracle_rds_cost + oracle_storage_cost
        
        # AWS Migration costs
        rec = recommendations[env]
        ec2_instance = rec['ec2']
        mongodb_cluster = rec['mongodb']
        
        # EC2 costs
        ec2_hourly_cost = pricing_data['ec2'].get(ec2_instance, 0.192)
        ec2_annual_cost = ec2_hourly_cost * specs['daily_usage'] * 365
        
        # EBS storage costs
        ebs_annual_cost = pricing_data['ebs']['gp3'] * specs['storage'] * 12
        
        # MongoDB Atlas costs
        mongodb_hourly_cost = pricing_data['mongodb_atlas'].get(mongodb_cluster, 0.54)
        mongodb_annual_cost = mongodb_hourly_cost * 24 * 365
        
        # Network and backup costs
        network_cost = 45.60 * 12  # NAT Gateway
        backup_cost = specs['storage'] * 0.05 * 12
        
        total_aws_cost = ec2_annual_cost + ebs_annual_cost + mongodb_annual_cost + network_cost + backup_cost
        
        # Calculate savings
        total_savings = current_total_cost - total_aws_cost
        savings_percentage = (total_savings / current_total_cost * 100) if current_total_cost > 0 else 0
        
        cost_data.append({
            'Environment': env,
            'Current_Oracle_License': oracle_license_per_env,
            'Current_Oracle_RDS': oracle_rds_cost,
            'Current_Oracle_Storage': oracle_storage_cost,
            'Current_Manpower': manpower_per_env,
            'Current_Total': current_total_cost,
            'AWS_EC2_Cost': ec2_annual_cost,
            'AWS_EBS_Cost': ebs_annual_cost,
            'AWS_MongoDB_Cost': mongodb_annual_cost,
            'AWS_Network_Cost': network_cost,
            'AWS_Backup_Cost': backup_cost,
            'AWS_Total_Cost': total_aws_cost,
            'Annual_Savings': total_savings,
            'Savings_Percentage': savings_percentage,
            'EC2_Instance': ec2_instance,
            'MongoDB_Cluster': mongodb_cluster
        })
    
    return pd.DataFrame(cost_data)

def calculate_migration_costs(data_size_tb, timeline_months, complexity_score):
    """Calculate migration-specific costs"""
    base_migration_cost = 15000  # Base monthly cost for migration team
    complexity_multiplier = 1 + (complexity_score / 100)
    monthly_migration_cost = base_migration_cost * complexity_multiplier
    total_migration_cost = monthly_migration_cost * timeline_months
    
    migration_data_transfer = data_size_tb * 1000 * 0.09  # $0.09 per GB
    tool_costs = 5000 * timeline_months
    training_costs = 10000
    
    return {
        'migration_team_cost': total_migration_cost,
        'data_transfer_cost': migration_data_transfer,
        'tool_costs': tool_costs,
        'training_costs': training_costs,
        'total_migration_cost': total_migration_cost + migration_data_transfer + tool_costs + training_costs
    }

def get_migration_strategy(complexity_score):
    """Determine migration strategy based on complexity"""
    if complexity_score < 25:
        return {
            'strategy': 'Rehost (Lift-and-Shift)',
            'description': 'Minimal changes required. Direct migration with basic refactoring.',
            'timeline': '2-4 months',
            'risk': 'Low',
            'effort': 'Low'
        }
    elif complexity_score < 50:
        return {
            'strategy': 'Refactor (Re-platform)',
            'description': 'Moderate application changes. Some PL/SQL conversion required.',
            'timeline': '4-8 months',
            'risk': 'Medium',
            'effort': 'Medium'
        }
    elif complexity_score < 75:
        return {
            'strategy': 'Revise (Re-architect)',
            'description': 'Significant application rework. Major PL/SQL conversion.',
            'timeline': '8-12 months',
            'risk': 'High',
            'effort': 'High'
        }
    else:
        return {
            'strategy': 'Rebuild (Rewrite)',
            'description': 'Complete application rewrite using cloud-native patterns.',
            'timeline': '12+ months',
            'risk': 'Very High',
            'effort': 'Very High'
        }

# Main Application
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Oracle to MongoDB Migration Analyzer</h1>', unsafe_allow_html=True)
    
    # Setup information
    if not ANTHROPIC_AVAILABLE:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **💡 To enable AI-powered analysis:**
        1. Install Anthropic: `pip install anthropic`
        2. Get API key from [Anthropic Console](https://console.anthropic.com/)
        3. Add to Streamlit secrets or environment variables
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AI Settings
        if ANTHROPIC_AVAILABLE:
            st.subheader("🤖 AI Analysis")
            enable_ai = st.checkbox("Enable AI Analysis", True)
            if enable_ai and 'anthropic_api_key' not in st.session_state:
                api_key = st.text_input("Anthropic API Key", type="password")
                if api_key:
                    st.session_state.anthropic_api_key = api_key
        else:
            enable_ai = False
            st.info("Install 'anthropic' package to enable AI features")
        
        st.markdown("---")
        
        # Configuration Form
        with st.form("migration_config"):
            st.subheader("🏢 Environment Configuration")
            
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
            
            analyze_button = st.form_submit_button("🔍 Analyze Migration", type="primary")
    
    # Main Analysis
    if analyze_button:
        with st.spinner('Analyzing migration requirements...'):
            # Get pricing data
            pricing_data = get_fallback_pricing_data()
            
            # Generate recommendations
            recommendations = recommend_instances(servers)
            
            # Calculate costs
            cost_df = calculate_costs(servers, pricing_data, oracle_license_cost, 
                                    manpower_cost, data_size_tb, recommendations)
            
            # Complexity analysis
            complexity_score, complexity_details, risk_factors = analyze_complexity(
                servers, data_size_tb, num_pl_sql_objects, num_applications
            )
            
            # Migration costs
            migration_costs = calculate_migration_costs(data_size_tb, migration_timeline, complexity_score)
            
            # Migration strategy
            strategy = get_migration_strategy(complexity_score)
        
        # Display Results
        st.success("✅ Analysis Complete!")
        
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
            savings_delta = f"{(total_annual_savings/total_current_cost*100):.1f}%" if total_current_cost > 0 else "0%"
            st.metric("Annual Savings", f"${total_annual_savings:,.0f}", delta=savings_delta)
        with col4:
            risk_level = "High" if complexity_score > 70 else "Medium" if complexity_score > 40 else "Low"
            st.metric("Complexity Score", f"{complexity_score}/100", delta=risk_level)
        
        # Migration Strategy
        st.header("📋 Recommended Migration Strategy")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Strategy:** {strategy['strategy']}")
            st.markdown(f"**Description:** {strategy['description']}")
            st.markdown(f"**Timeline:** {strategy['timeline']}")
        
        with col2:
            st.markdown(f"**Risk Level:** {strategy['risk']}")
            st.markdown(f"**Effort Level:** {strategy['effort']}")
        
        # Risk Assessment
        if risk_factors:
            st.header("⚠️ Risk Assessment")
            st.markdown('<div class="risk-assessment">', unsafe_allow_html=True)
            for risk in risk_factors:
                st.markdown(f"• {risk}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # AI Analysis (if available)
        if enable_ai and ANTHROPIC_AVAILABLE:
            st.header("🤖 AI-Powered Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Migration Analysis", "PL/SQL Strategy", 
                "Security & Compliance", "Performance Optimization"
            ])
            
            with tab1:
                st.markdown('<div class="ai-insight">', unsafe_allow_html=True)
                migration_prompt = f"""
                Analyze this Oracle to MongoDB migration:
                - Environments: {list(servers.keys())}
                - Data Size: {data_size_tb} TB
                - Complexity: {complexity_score}/100
                - PL/SQL Objects: {num_pl_sql_objects}
                - Timeline: {migration_timeline} months
                
                Provide migration feasibility, challenges, and recommendations.
                """
                st.markdown(get_claude_analysis(migration_prompt, "migration"))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="ai-insight">', unsafe_allow_html=True)
                plsql_prompt = f"""
                PL/SQL conversion strategy for {num_pl_sql_objects} objects.
                Complexity score: {complexity_score}/100.
                Provide conversion approach and MongoDB alternatives.
                """
                st.markdown(get_claude_analysis(plsql_prompt, "plsql"))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="ai-insight">', unsafe_allow_html=True)
                security_prompt = f"""
                Security analysis for Oracle to MongoDB migration.
                Production specs: {servers.get('Prod', 'N/A')}
                High availability: {high_availability}
                Provide security architecture and compliance recommendations.
                """
                st.markdown(get_claude_analysis(security_prompt, "security"))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown('<div class="ai-insight">', unsafe_allow_html=True)
                perf_prompt = f"""
                Performance optimization for MongoDB migration.
                Data size: {data_size_tb} TB
                Production specs: {servers.get('Prod', 'N/A')}
                Provide schema design and optimization recommendations.
                """
                st.markdown(get_claude_analysis(perf_prompt, "performance"))
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Cost Analysis
        st.header("💰 Cost Analysis")
        
        # Cost breakdown chart
        fig = go.Figure()
        
        environments = cost_df['Environment']
        current_costs = cost_df['Current_Total']
        aws_costs = cost_df['AWS_Total_Cost']
        
        fig.add_trace(go.Bar(
            name='Current Oracle Costs',
            x=environments,
            y=current_costs,
            marker_color='#FF6B6B'
        ))
        
        fig.add_trace(go.Bar(
            name='Projected AWS Costs',
            x=environments,
            y=aws_costs,
            marker_color='#4ECDC4'
        ))
        
        fig.update_layout(
            title='Cost Comparison by Environment',
            xaxis_title='Environment',
            yaxis_title='Annual Cost ($)',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed cost table
        st.subheader("Detailed Cost Breakdown")
        
        # Format currency columns
        currency_cols = [col for col in cost_df.columns if 'Cost' in col or 'Savings' in col]
        formatted_df = cost_df.copy()
        for col in currency_cols:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.0f}")
        
        formatted_df['Savings_Percentage'] = cost_df['Savings_Percentage'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(formatted_df, use_container_width=True)
        
        # Migration Cost Analysis
        st.header("🚚 Migration Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Migration Costs")
            st.metric("Migration Team", f"${migration_costs['migration_team_cost']:,.0f}")
            st.metric("Data Transfer", f"${migration_costs['data_transfer_cost']:,.0f}")
            st.metric("Tools & Licenses", f"${migration_costs['tool_costs']:,.0f}")
            st.metric("Training", f"${migration_costs['training_costs']:,.0f}")
            st.metric("**Total Migration Cost**", f"${migration_costs['total_migration_cost']:,.0f}")
        
        with col2:
            st.subheader("ROI Analysis")
            payback_months = migration_costs['total_migration_cost'] / (total_annual_savings / 12) if total_annual_savings > 0 else float('inf')
            three_year_roi = (total_annual_savings * 3 - migration_costs['total_migration_cost']) / migration_costs['total_migration_cost'] * 100 if migration_costs['total_migration_cost'] > 0 else 0
            
            st.metric("Payback Period", f"{payback_months:.1f} months" if payback_months != float('inf') else "No payback")
            st.metric("3-Year ROI", f"{three_year_roi:.1f}%")
            st.metric("Break-even", "Yes" if payback_months <= 36 else "No")
        
        # ROI Timeline Chart
        roi_years = []
        cumulative_savings = []
        cumulative_investment = migration_costs['total_migration_cost']
        
        for year in range(1, 4):
            annual_savings = total_annual_savings * (1 + 0.05 * (year - 1))  # 5% improvement per year
            if year == 1:
                net_savings = annual_savings - migration_costs['total_migration_cost']
            else:
                net_savings = annual_savings
            
            cumulative_savings.append(sum([total_annual_savings * (1 + 0.05 * (y - 1)) for y in range(1, year + 1)]) - migration_costs['total_migration_cost'])
            roi_years.append(f"Year {year}")
        
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=roi_years,
            y=cumulative_savings,
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=8)
        ))
        
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Break-even Point")
        
        fig_roi.update_layout(
            title='3-Year ROI Timeline',
            xaxis_title='Timeline',
            yaxis_title='Cumulative Savings ($)',
            height=400
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)
        
        # Instance Recommendations
        st.header("💻 Infrastructure Recommendations")
        
        recommendations_df = pd.DataFrame([
            {
                'Environment': env,
                'Current Oracle Config': f"{servers[env]['cpu']} CPU, {servers[env]['ram']}GB RAM, {servers[env]['storage']}GB Storage",
                'Recommended EC2': rec['ec2'],
                'Recommended MongoDB': rec['mongodb'],
                'Daily Usage': f"{servers[env]['daily_usage']} hours"
            }
            for env, rec in recommendations.items()
        ])
        
        st.dataframe(recommendations_df, use_container_width=True)
        
        # Migration Timeline
        st.header("📅 Migration Timeline & Phases")
        
        phases = [
            {"Phase": "Discovery & Assessment", "Duration": "2-4 weeks", "Key Activities": "Inventory, complexity analysis, requirements gathering"},
            {"Phase": "Architecture Design", "Duration": "2-3 weeks", "Key Activities": "Schema design, infrastructure planning, security architecture"},
            {"Phase": "Development Setup", "Duration": "3-4 weeks", "Key Activities": "Environment setup, tooling, initial prototypes"},
            {"Phase": "Schema Migration", "Duration": "4-6 weeks", "Key Activities": "Schema conversion, data modeling, index design"},
            {"Phase": "Application Refactoring", "Duration": f"{max(4, num_pl_sql_objects // 100)}-{max(8, num_pl_sql_objects // 50)} weeks", "Key Activities": "PL/SQL conversion, application updates, integration changes"},
            {"Phase": "Testing & Validation", "Duration": "3-4 weeks", "Key Activities": "Unit testing, integration testing, performance testing"},
            {"Phase": "Data Migration", "Duration": "1-2 weeks", "Key Activities": "Data transfer, validation, synchronization"},
            {"Phase": "Go-Live & Optimization", "Duration": "2-3 weeks", "Key Activities": "Production cutover, monitoring, performance tuning"}
        ]
        
        phases_df = pd.DataFrame(phases)
        st.dataframe(phases_df, use_container_width=True)
        
        # Complexity Breakdown
        st.header("🎯 Complexity Analysis")
        
        complexity_fig = go.Figure(data=[
            go.Bar(
                x=list(complexity_details.keys()),
                y=list(complexity_details.values()),
                marker_color=['#FF9999' if v > 70 else '#FFCC99' if v > 40 else '#99FF99' for v in complexity_details.values()]
            )
        ])
        
        complexity_fig.update_layout(
            title='Complexity Factors Breakdown',
            xaxis_title='Factors',
            yaxis_title='Complexity Score (0-100)',
            height=400
        )
        
        st.plotly_chart(complexity_fig, use_container_width=True)
        
        # Action Items & Next Steps
        st.header("✅ Recommended Next Steps")
        
        next_steps = [
            "🔍 **Conduct Detailed Assessment** - Perform comprehensive discovery of Oracle databases, applications, and dependencies",
            "📋 **Create Migration Plan** - Develop detailed project plan with timelines, resources, and milestones",
            "👥 **Assemble Migration Team** - Identify and train team members for Oracle and MongoDB technologies",
            "🏗️ **Setup Development Environment** - Create development and testing environments for migration proof of concept",
            "📊 **Pilot Migration** - Start with least critical environment to validate approach and refine processes",
            "🔧 **Tool Selection** - Evaluate and select migration tools, monitoring solutions, and development frameworks",
            "📚 **Training Program** - Implement comprehensive training for development and operations teams",
            "🛡️ **Security Review** - Conduct security assessment and implement necessary compliance measures"
        ]
        
        for step in next_steps:
            st.markdown(f"- {step}")
        
        # Download Options
        st.header("📄 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Cost analysis CSV
            csv_data = cost_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Cost Analysis (CSV)",
                data=csv_data,
                file_name=f"migration_cost_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Recommendations JSON
            recommendations_json = json.dumps({
                'servers': servers,
                'recommendations': recommendations,
                'complexity_score': complexity_score,
                'strategy': strategy,
                'migration_costs': migration_costs,
                'total_savings': float(total_annual_savings)
            }, indent=2)
            
            st.download_button(
                label="⚙️ Download Configuration (JSON)",
                data=recommendations_json,
                file_name=f"migration_config_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col3:
            # Summary report
            summary_report = f"""
Oracle to MongoDB Migration Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
=================
Current Annual Cost: ${total_current_cost:,.0f}
Projected AWS Cost: ${total_aws_cost:,.0f}
Annual Savings: ${total_annual_savings:,.0f}
Savings Percentage: {(total_annual_savings/total_current_cost*100):.1f}%

COMPLEXITY ANALYSIS
===================
Overall Complexity Score: {complexity_score}/100
Recommended Strategy: {strategy['strategy']}
Estimated Timeline: {strategy['timeline']}
Risk Level: {strategy['risk']}

MIGRATION COSTS
===============
Total Migration Cost: ${migration_costs['total_migration_cost']:,.0f}
Payback Period: {payback_months:.1f} months
3-Year ROI: {three_year_roi:.1f}%

ENVIRONMENTS
============
{chr(10).join([f"{env}: {specs['cpu']} CPU, {specs['ram']}GB RAM, {specs['storage']}GB Storage" for env, specs in servers.items()])}

RECOMMENDATIONS
===============
{chr(10).join([f"{env}: {rec['ec2']} (EC2) + {rec['mongodb']} (MongoDB)" for env, rec in recommendations.items()])}
"""
            
            st.download_button(
                label="📋 Download Summary Report (TXT)",
                data=summary_report,
                file_name=f"migration_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

# Installation and Setup Instructions
def show_setup_instructions():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Setup Instructions")
    
    with st.sidebar.expander("📦 Installation"):
        st.markdown("""
        **Required packages:**
        ```bash
        pip install streamlit pandas plotly
        ```
        
        **Optional (for AI features):**
        ```bash
        pip install anthropic
        ```
        
        **Optional (for AWS pricing):**
        ```bash
        pip install boto3
        ```
        """)
    
    with st.sidebar.expander("🤖 AI Setup"):
        st.markdown("""
        **1. Get Anthropic API Key:**
        - Visit [console.anthropic.com](https://console.anthropic.com/)
        - Create account and generate API key
        
        **2. Configure API Key:**
        - Set environment variable:
          ```bash
          export ANTHROPIC_API_KEY=your_key_here
          ```
        - Or add to `.streamlit/secrets.toml`:
          ```toml
          ANTHROPIC_API_KEY = "your_key_here"
          ```
        """)
    
    with st.sidebar.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Dependency conflicts:**
        - Use virtual environment: `python -m venv venv`
        - Update pip: `pip install --upgrade pip`
        - Install individually: `pip install package_name`
        
        **AWS API issues:**
        - Check AWS credentials
        - Verify region access
        - Use fallback pricing if needed
        """)

if __name__ == "__main__":
    show_setup_instructions()
    main()