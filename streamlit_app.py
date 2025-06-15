import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from io import BytesIO
import base64

# Page Configuration
st.set_page_config(
    page_title="Enterprise Oracle to MongoDB Migration Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏢"
)

# Enhanced Enterprise CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .enterprise-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .enterprise-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .enterprise-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .config-section {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .config-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #4299e1;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #38b2ac;
        text-align: center;
        margin: 1rem 0;
    }
    
    .analysis-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        margin: 1rem 0;
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        margin: 3rem 0;
        border-radius: 1px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'servers_config' not in st.session_state:
        st.session_state.servers_config = {}
    if 'migration_params' not in st.session_state:
        st.session_state.migration_params = {}

def handle_bulk_upload():
    """Enhanced bulk upload functionality"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Bulk Configuration Upload")
    st.markdown("Upload CSV, Excel, or JSON files with your environment configurations")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'xlsx', 'json'],
        help="Supported formats: CSV, Excel (.xlsx), JSON"
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            with st.spinner(f'Processing {file_extension.upper()} file...'):
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    st.success("✅ CSV file loaded successfully!")
                    
                elif file_extension == 'xlsx':
                    df = pd.read_excel(uploaded_file)
                    st.success("✅ Excel file loaded successfully!")
                    
                elif file_extension == 'json':
                    config_data = json.load(uploaded_file)
                    st.success("✅ JSON configuration loaded successfully!")
                    st.json(config_data)
                    st.session_state.servers_config = config_data
                    st.markdown('</div>', unsafe_allow_html=True)
                    return config_data
                
                # Display preview for CSV/Excel
                if file_extension in ['csv', 'xlsx']:
                    st.markdown("#### 👀 File Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Process the file
                    servers_config = process_environment_file(df)
                    if servers_config:
                        st.session_state.servers_config = servers_config
                        return servers_config
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.markdown("**Expected file format:**")
            st.markdown("- **CSV/Excel:** Columns: Environment, CPU, RAM, Storage, Daily_Usage (optional), Throughput (optional)")
            st.markdown("- **JSON:** Nested structure with environment configurations")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def process_environment_file(df):
    """Process uploaded environment configuration file"""
    try:
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Check for required columns
        required_columns = ['environment', 'cpu', 'ram', 'storage']
        available_columns = df.columns.tolist()
        
        # Try to map common column variations
        column_mapping = {
            'env': 'environment',
            'env_name': 'environment',
            'environment_name': 'environment',
            'vcpu': 'cpu',
            'cores': 'cpu',
            'memory': 'ram',
            'ram_gb': 'ram',
            'memory_gb': 'ram',
            'storage_gb': 'storage',
            'disk': 'storage',
            'disk_gb': 'storage'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in available_columns:
                df = df.rename(columns={old_col: new_col})
        
        # Check for missing columns after mapping
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.markdown("**Available columns:** " + ", ".join(df.columns.tolist()))
            return None
        
        # Set defaults for optional columns
        if 'daily_usage' not in df.columns:
            df['daily_usage'] = 20  # Default 20 hours per day
        if 'throughput' not in df.columns:
            df['throughput'] = df['cpu'] * 1000  # Default: 1000 IOPS per CPU
        
        # Convert to the expected format
        servers = {}
        for idx, row in df.iterrows():
            try:
                env_name = str(row['environment']).strip()
                if not env_name or env_name.lower() in ['nan', 'none', '']:
                    continue
                    
                servers[env_name] = {
                    'cpu': int(float(row['cpu'])),
                    'ram': int(float(row['ram'])),
                    'storage': int(float(row['storage'])),
                    'daily_usage': int(float(row.get('daily_usage', 20))),
                    'throughput': int(float(row.get('throughput', row['cpu'] * 1000)))
                }
            except (ValueError, TypeError) as e:
                st.warning(f"⚠️ Skipping row {idx + 1} due to invalid data: {str(e)}")
                continue
        
        if not servers:
            st.error("❌ No valid environment configurations found in the file")
            return None
        
        st.success(f"✅ Successfully processed {len(servers)} environments!")
        
        # Display processed environments
        st.markdown("#### 📊 Processed Environments")
        processed_df = pd.DataFrame.from_dict(servers, orient='index')
        processed_df.index.name = 'Environment'
        st.dataframe(processed_df, use_container_width=True)
        
        return servers
        
    except Exception as e:
        st.error(f"❌ Error processing environment data: {str(e)}")
        return None

def create_sample_template():
    """Create and display sample template for bulk upload"""
    st.markdown("#### 📋 Sample Template")
    
    sample_data = {
        'Environment': ['Development', 'QA', 'Staging', 'Production'],
        'CPU': [4, 8, 16, 32],
        'RAM': [16, 32, 64, 128],
        'Storage': [200, 500, 1000, 2000],
        'Daily_Usage': [12, 16, 20, 24],
        'Throughput': [4000, 8000, 16000, 32000]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    
    # Download template
    csv_template = sample_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="migration_template.csv",
        mime="text/csv"
    )

def manual_environment_setup():
    """Enhanced manual environment setup"""
    st.markdown("### 🔧 Manual Environment Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=3)
    with col2:
        if st.button("🔄 Reset Configuration"):
            st.session_state.servers_config = {}
            st.rerun()
    
    servers = {}
    default_envs = ['Development', 'QA', 'Production', 'Staging', 'UAT']
    
    # Create environment configuration in columns
    if num_environments <= 3:
        cols = st.columns(num_environments)
    else:
        cols = st.columns(3)
    
    for i in range(num_environments):
        col_idx = i % len(cols)
        with cols[col_idx]:
            with st.expander(f"📍 Environment {i+1}", expanded=True):
                env_name = st.text_input(
                    "Environment Name", 
                    value=default_envs[i] if i < len(default_envs) else f"Environment_{i+1}",
                    key=f"env_name_{i}"
                )
                
                # Resource configuration
                cpu = st.number_input(
                    "CPU Cores", 
                    min_value=1, max_value=128, 
                    value=[4, 8, 32][min(i, 2)], 
                    key=f"cpu_{i}"
                )
                
                ram = st.number_input(
                    "RAM (GB)", 
                    min_value=4, max_value=1024, 
                    value=[16, 32, 128][min(i, 2)], 
                    key=f"ram_{i}"
                )
                
                storage = st.number_input(
                    "Storage (GB)", 
                    min_value=10, max_value=10000, 
                    value=[200, 500, 2000][min(i, 2)], 
                    key=f"storage_{i}"
                )
                
                throughput = st.number_input(
                    "IOPS", 
                    min_value=100, max_value=100000, 
                    value=[2000, 5000, 20000][min(i, 2)], 
                    key=f"throughput_{i}"
                )
                
                daily_usage = st.slider(
                    "Daily Usage (Hours)", 
                    min_value=1, max_value=24, 
                    value=[12, 16, 24][min(i, 2)], 
                    key=f"usage_{i}"
                )
                
                servers[env_name] = {
                    'cpu': cpu,
                    'ram': ram,
                    'storage': storage,
                    'throughput': throughput,
                    'daily_usage': daily_usage
                }
    
    # Save configuration
    if servers:
        st.session_state.servers_config = servers
        
        # Display summary
        st.markdown("#### 📊 Configuration Summary")
        summary_df = pd.DataFrame.from_dict(servers, orient='index')
        summary_df.index.name = 'Environment'
        st.dataframe(summary_df, use_container_width=True)
    
    return servers

def get_migration_parameters():
    """Get migration parameters"""
    st.markdown("### 📋 Migration Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💾 Data Configuration")
        data_size_tb = st.number_input("Total Data Size (TB)", min_value=1, max_value=1000, value=25)
        num_pl_sql_objects = st.number_input("PL/SQL Objects", min_value=0, max_value=10000, value=500)
        num_applications = st.number_input("Connected Applications", min_value=1, max_value=100, value=5)
    
    with col2:
        st.markdown("#### ⏱️ Timeline & Network")
        migration_timeline = st.slider("Migration Timeline (Months)", min_value=3, max_value=24, value=8)
        use_direct_connect = st.checkbox("Use AWS Direct Connect", value=True)
        bandwidth_option = st.selectbox("Bandwidth", ["1 Gbps", "10 Gbps", "100 Gbps"], index=1)
        bandwidth_gbps = int(bandwidth_option.split()[0])
    
    with col3:
        st.markdown("#### 💰 Cost Parameters")
        oracle_license_cost = st.number_input("Oracle License Cost ($/year)", min_value=0, value=150000)
        manpower_cost = st.number_input("Maintenance & Support ($/year)", min_value=0, value=200000)
        migration_budget = st.number_input("Migration Budget ($)", min_value=0, value=500000)
    
    # Store parameters
    params = {
        'data_size_tb': data_size_tb,
        'num_pl_sql_objects': num_pl_sql_objects,
        'num_applications': num_applications,
        'migration_timeline': migration_timeline,
        'use_direct_connect': use_direct_connect,
        'bandwidth_gbps': bandwidth_gbps,
        'oracle_license_cost': oracle_license_cost,
        'manpower_cost': manpower_cost,
        'migration_budget': migration_budget
    }
    
    st.session_state.migration_params = params
    return params

def analyze_workload(servers, params):
    """Simplified workload analysis"""
    
    with st.spinner('🔄 Analyzing workload and calculating costs...'):
        progress = st.progress(0)
        
        # Step 1: Instance Recommendations
        progress.progress(25)
        recommendations = get_instance_recommendations(servers)
        
        # Step 2: Cost Analysis
        progress.progress(50)
        cost_analysis = calculate_cost_analysis(servers, params, recommendations)
        
        # Step 3: Complexity Analysis
        progress.progress(75)
        complexity_analysis = calculate_complexity_analysis(servers, params)
        
        # Step 4: Migration Strategy
        progress.progress(100)
        migration_strategy = get_migration_strategy(complexity_analysis['score'])
        
        # Compile results
        results = {
            'servers': servers,
            'params': params,
            'recommendations': recommendations,
            'cost_analysis': cost_analysis,
            'complexity_analysis': complexity_analysis,
            'migration_strategy': migration_strategy,
            'timestamp': datetime.now()
        }
        
        st.session_state.analysis_results = results
        progress.empty()
    
    return results

def get_instance_recommendations(servers):
    """Get AWS instance recommendations"""
    recommendations = {}
    
    for env, specs in servers.items():
        cpu = specs['cpu']
        ram = specs['ram']
        
        # EC2 Recommendations
        if cpu <= 2 and ram <= 8:
            ec2_instance = 't3.medium'
        elif cpu <= 4 and ram <= 16:
            ec2_instance = 't3.large'
        elif cpu <= 8 and ram <= 32:
            ec2_instance = 'm5.xlarge'
        elif cpu <= 16 and ram <= 64:
            ec2_instance = 'r5.2xlarge'
        elif cpu <= 32 and ram <= 128:
            ec2_instance = 'r5.4xlarge'
        else:
            ec2_instance = 'r5.8xlarge'
        
        # MongoDB Atlas Recommendations
        if env.lower() in ['dev', 'development']:
            mongodb_cluster = 'M10'
        elif env.lower() in ['qa', 'test']:
            mongodb_cluster = 'M20'
        elif env.lower() in ['uat', 'staging']:
            mongodb_cluster = 'M30'
        else:  # Production
            if cpu <= 8:
                mongodb_cluster = 'M40'
            elif cpu <= 16:
                mongodb_cluster = 'M60'
            elif cpu <= 32:
                mongodb_cluster = 'M80'
            else:
                mongodb_cluster = 'M140'
        
        recommendations[env] = {
            'ec2_instance': ec2_instance,
            'mongodb_cluster': mongodb_cluster
        }
    
    return recommendations

def calculate_cost_analysis(servers, params, recommendations):
    """Calculate cost analysis"""
    
    # Simplified pricing data
    ec2_pricing = {
        't3.medium': 0.0416, 't3.large': 0.0832, 't3.xlarge': 0.1664,
        'm5.large': 0.096, 'm5.xlarge': 0.192, 'm5.2xlarge': 0.384,
        'r5.large': 0.126, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504,
        'r5.4xlarge': 1.008, 'r5.8xlarge': 2.016
    }
    
    mongodb_pricing = {
        'M10': 0.08, 'M20': 0.12, 'M30': 0.54, 'M40': 1.08,
        'M50': 2.16, 'M60': 4.32, 'M80': 8.64, 'M140': 17.28
    }
    
    # Storage and other costs (per GB per month)
    storage_cost_per_gb = 0.08
    backup_cost_per_gb = 0.05
    
    cost_data = []
    total_current_cost = 0
    total_aws_cost = 0
    
    oracle_license_per_env = params['oracle_license_cost'] / len(servers)
    manpower_per_env = params['manpower_cost'] / len(servers)
    
    for env, specs in servers.items():
        # Current Oracle costs
        current_infrastructure = 500 * specs['cpu']  # Estimated Oracle infrastructure cost
        current_total = oracle_license_per_env + manpower_per_env + current_infrastructure
        
        # AWS costs
        rec = recommendations[env]
        ec2_hourly = ec2_pricing.get(rec['ec2_instance'], 0.192)
        mongodb_hourly = mongodb_pricing.get(rec['mongodb_cluster'], 0.54)
        
        # Annual costs
        ec2_annual = ec2_hourly * specs['daily_usage'] * 365
        mongodb_annual = mongodb_hourly * 24 * 365
        storage_annual = specs['storage'] * storage_cost_per_gb * 12
        backup_annual = specs['storage'] * backup_cost_per_gb * 12
        
        aws_total = ec2_annual + mongodb_annual + storage_annual + backup_annual
        
        # Calculate savings
        annual_savings = current_total - aws_total
        savings_percentage = (annual_savings / current_total * 100) if current_total > 0 else 0
        
        cost_data.append({
            'Environment': env,
            'Current_Total': current_total,
            'AWS_Total': aws_total,
            'Annual_Savings': annual_savings,
            'Savings_Percentage': savings_percentage,
            'EC2_Instance': rec['ec2_instance'],
            'MongoDB_Cluster': rec['mongodb_cluster']
        })
        
        total_current_cost += current_total
        total_aws_cost += aws_total
    
    # Migration costs
    data_size_tb = params['data_size_tb']
    timeline_months = params['migration_timeline']
    
    migration_team_cost = 25000 * timeline_months
    data_transfer_cost = data_size_tb * 1000 * 0.02 if params['use_direct_connect'] else data_size_tb * 100
    tools_and_training = 50000
    aws_services = 20000
    contingency = (migration_team_cost + data_transfer_cost + tools_and_training + aws_services) * 0.15
    
    total_migration_cost = migration_team_cost + data_transfer_cost + tools_and_training + aws_services + contingency
    
    return {
        'cost_breakdown': pd.DataFrame(cost_data),
        'total_current_cost': total_current_cost,
        'total_aws_cost': total_aws_cost,
        'total_annual_savings': total_current_cost - total_aws_cost,
        'migration_costs': {
            'migration_team': migration_team_cost,
            'data_transfer': data_transfer_cost,
            'tools_training': tools_and_training,
            'aws_services': aws_services,
            'contingency': contingency,
            'total': total_migration_cost
        }
    }

def calculate_complexity_analysis(servers, params):
    """Calculate migration complexity"""
    
    # Get production specs (use last environment as fallback)
    prod_specs = list(servers.values())[-1]
    
    # Complexity factors
    factors = {
        'Infrastructure Size': min(100, prod_specs['cpu'] * 3),
        'Data Volume': min(100, params['data_size_tb'] * 2),
        'PL/SQL Complexity': min(100, params['num_pl_sql_objects'] / 10),
        'Application Integration': min(100, params['num_applications'] * 15),
        'Environment Count': min(100, len(servers) * 20)
    }
    
    # Calculate weighted score
    weights = {
        'Infrastructure Size': 0.2,
        'Data Volume': 0.25,
        'PL/SQL Complexity': 0.25,
        'Application Integration': 0.2,
        'Environment Count': 0.1
    }
    
    total_score = sum(factors[factor] * weights[factor] for factor in factors)
    
    # Risk factors
    risk_factors = []
    if factors['PL/SQL Complexity'] > 60:
        risk_factors.append("High PL/SQL complexity requires extensive refactoring")
    if factors['Data Volume'] > 70:
        risk_factors.append("Large data volume may require extended migration timeline")
    if factors['Application Integration'] > 70:
        risk_factors.append("Multiple application dependencies increase integration complexity")
    if len(servers) > 3:
        risk_factors.append("Multiple environments require coordinated migration approach")
    
    return {
        'score': int(total_score),
        'factors': factors,
        'risk_factors': risk_factors
    }

def get_migration_strategy(complexity_score):
    """Get migration strategy based on complexity"""
    if complexity_score < 30:
        return {
            'strategy': 'Lift and Shift',
            'timeline': '3-5 months',
            'risk': 'Low',
            'description': 'Direct migration with minimal changes'
        }
    elif complexity_score < 50:
        return {
            'strategy': 'Re-platform',
            'timeline': '5-8 months',
            'risk': 'Medium',
            'description': 'Migration with some optimization and refactoring'
        }
    elif complexity_score < 70:
        return {
            'strategy': 'Re-architect',
            'timeline': '8-12 months',
            'risk': 'Medium-High',
            'description': 'Significant redesign for cloud-native architecture'
        }
    else:
        return {
            'strategy': 'Transform',
            'timeline': '12+ months',
            'risk': 'High',
            'description': 'Complete transformation with modern architecture patterns'
        }

def display_results(results):
    """Display analysis results"""
    if not results:
        st.info("👆 Please complete the analysis first")
        return
    
    cost_analysis = results['cost_analysis']
    complexity_analysis = results['complexity_analysis']
    migration_strategy = results['migration_strategy']
    
    # Executive Summary
    st.markdown("## 📊 Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annual Savings", f"${cost_analysis['total_annual_savings']:,.0f}")
    
    with col2:
        st.metric("Migration Cost", f"${cost_analysis['migration_costs']['total']:,.0f}")
    
    with col3:
        st.metric("Complexity Score", f"{complexity_analysis['score']}/100")
    
    with col4:
        roi = ((cost_analysis['total_annual_savings'] * 3 - cost_analysis['migration_costs']['total']) / 
               cost_analysis['migration_costs']['total'] * 100)
        st.metric("3-Year ROI", f"{roi:.1f}%")
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Cost Analysis", "📈 Visualizations", "🎯 Strategy", "📋 Export"])
    
    with tab1:
        st.markdown("### Environment Cost Breakdown")
        st.dataframe(cost_analysis['cost_breakdown'], use_container_width=True)
        
        st.markdown("### Migration Cost Breakdown")
        migration_df = pd.DataFrame([
            {'Component': k.replace('_', ' ').title(), 'Cost': f"${v:,.0f}"}
            for k, v in cost_analysis['migration_costs'].items()
        ])
        st.dataframe(migration_df, use_container_width=True)
    
    with tab2:
        # Cost comparison chart
        fig = go.Figure()
        
        envs = cost_analysis['cost_breakdown']['Environment'].tolist()
        current_costs = cost_analysis['cost_breakdown']['Current_Total'].tolist()
        aws_costs = cost_analysis['cost_breakdown']['AWS_Total'].tolist()
        
        fig.add_trace(go.Bar(name='Current Oracle', x=envs, y=current_costs, marker_color='lightcoral'))
        fig.add_trace(go.Bar(name='Projected AWS', x=envs, y=aws_costs, marker_color='lightblue'))
        
        fig.update_layout(
            title='Cost Comparison by Environment',
            xaxis_title='Environment',
            yaxis_title='Annual Cost ($)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Complexity factors
        fig2 = go.Figure()
        
        factors = list(complexity_analysis['factors'].keys())
        values = list(complexity_analysis['factors'].values())
        
        fig2.add_trace(go.Bar(x=factors, y=values, marker_color='lightgreen'))
        fig2.update_layout(
            title='Migration Complexity Factors',
            xaxis_title='Factor',
            yaxis_title='Complexity Score',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown("### Migration Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Strategy:** {migration_strategy['strategy']}
            
            **Timeline:** {migration_strategy['timeline']}
            
            **Risk Level:** {migration_strategy['risk']}
            
            **Description:** {migration_strategy['description']}
            """)
        
        with col2:
            st.markdown("**Risk Factors:**")
            for risk in complexity_analysis['risk_factors']:
                st.markdown(f"• {risk}")
        
        st.markdown("### Technical Recommendations")
        recommendations_df = pd.DataFrame([
            {
                'Environment': env,
                'Current Config': f"{specs['cpu']} vCPU, {specs['ram']} GB RAM",
                'Recommended EC2': results['recommendations'][env]['ec2_instance'],
                'Recommended MongoDB': results['recommendations'][env]['mongodb_cluster']
            }
            for env, specs in results['servers'].items()
        ])
        st.dataframe(recommendations_df, use_container_width=True)
    
    with tab4:
        st.markdown("### Export Results")
        
        # Generate text report
        text_report = generate_text_report(results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Text Report",
                data=text_report,
                file_name=f"Migration_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            csv_data = cost_analysis['cost_breakdown'].to_csv(index=False)
            st.download_button(
                label="📊 Download Cost Analysis (CSV)",
                data=csv_data,
                file_name=f"Cost_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def generate_text_report(results):
    """Generate comprehensive text report"""
    cost_analysis = results['cost_analysis']
    complexity_analysis = results['complexity_analysis']
    migration_strategy = results['migration_strategy']
    params = results['params']
    
    total_savings = cost_analysis['total_annual_savings']
    migration_cost = cost_analysis['migration_costs']['total']
    roi = ((total_savings * 3 - migration_cost) / migration_cost * 100) if migration_cost > 0 else 0
    
    return f"""
ORACLE TO MONGODB MIGRATION ANALYSIS REPORT
==========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Annual Cost Savings: ${total_savings:,.0f}
Migration Investment: ${migration_cost:,.0f}
3-Year ROI: {roi:.1f}%
Complexity Score: {complexity_analysis['score']}/100
Recommended Strategy: {migration_strategy['strategy']}

ENVIRONMENT ANALYSIS
===================
{chr(10).join([f"{env}: {specs['cpu']} vCPU, {specs['ram']} GB RAM, {specs['storage']} GB Storage" 
               for env, specs in results['servers'].items()])}

MIGRATION PARAMETERS
===================
Data Size: {params['data_size_tb']} TB
Timeline: {params['migration_timeline']} months
PL/SQL Objects: {params['num_pl_sql_objects']:,}
Connected Applications: {params['num_applications']}
Oracle License Cost: ${params['oracle_license_cost']:,}/year
Maintenance Cost: ${params['manpower_cost']:,}/year

COST BREAKDOWN
=============
Current Oracle Total: ${cost_analysis['total_current_cost']:,.0f}/year
Projected AWS Total: ${cost_analysis['total_aws_cost']:,.0f}/year
Annual Savings: ${total_savings:,.0f}

Migration Costs:
• Migration Team: ${cost_analysis['migration_costs']['migration_team']:,.0f}
• Data Transfer: ${cost_analysis['migration_costs']['data_transfer']:,.0f}
• Tools & Training: ${cost_analysis['migration_costs']['tools_training']:,.0f}
• AWS Services: ${cost_analysis['migration_costs']['aws_services']:,.0f}
• Contingency: ${cost_analysis['migration_costs']['contingency']:,.0f}
• Total: ${migration_cost:,.0f}

COMPLEXITY ANALYSIS
==================
Overall Score: {complexity_analysis['score']}/100

Factor Breakdown:
{chr(10).join([f"• {factor}: {score:.0f}/100" for factor, score in complexity_analysis['factors'].items()])}

Risk Factors:
{chr(10).join([f"• {risk}" for risk in complexity_analysis['risk_factors']])}

MIGRATION STRATEGY
=================
Strategy: {migration_strategy['strategy']}
Timeline: {migration_strategy['timeline']}
Risk Level: {migration_strategy['risk']}
Description: {migration_strategy['description']}

RECOMMENDATIONS
==============
{chr(10).join([f"{env}: {results['recommendations'][env]['ec2_instance']} + {results['recommendations'][env]['mongodb_cluster']}" 
               for env in results['servers'].keys()])}

NEXT STEPS
==========
1. Secure stakeholder approval and budget allocation
2. Form migration team and begin training
3. Set up AWS infrastructure and MongoDB Atlas
4. Execute migration following {migration_strategy['strategy']} approach
5. Validate data integrity and application functionality
6. Go-live and post-migration optimization
"""

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="enterprise-header">
        <h1>🏢 Enterprise Oracle to MongoDB Migration Analyzer</h1>
        <p>Comprehensive analysis and planning for enterprise database migration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Select Section:",
            ["🔧 Configuration", "🚀 Analysis", "📊 Results"],
            key="main_navigation"
        )
        
        # Quick status
        if st.session_state.servers_config:
            st.success(f"✅ {len(st.session_state.servers_config)} environments configured")
        else:
            st.warning("⚠️ No environments configured")
        
        if st.session_state.analysis_results:
            st.success("✅ Analysis complete")
        else:
            st.info("ℹ️ Analysis pending")
    
    # Main content based on selected page
    if page == "🔧 Configuration":
        st.markdown("## 🔧 Environment Configuration")
        
        # Configuration method selection
        config_method = st.radio(
            "Choose configuration method:",
            ["📝 Manual Entry", "📁 Bulk Upload"],
            horizontal=True,
            key="config_method"
        )
        
        if config_method == "📁 Bulk Upload":
            st.markdown("### 📁 Bulk Upload Configuration")
            
            # Show sample template first
            with st.expander("📋 View Sample Template", expanded=False):
                create_sample_template()
            
            # File upload
            servers_config = handle_bulk_upload()
            
            if not servers_config and st.session_state.servers_config:
                st.success("✅ Using previously uploaded configuration")
                st.dataframe(
                    pd.DataFrame.from_dict(st.session_state.servers_config, orient='index'),
                    use_container_width=True
                )
        
        else:  # Manual Entry
            servers_config = manual_environment_setup()
        
        # Migration parameters (always show)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        migration_params = get_migration_parameters()
        
        # Show configuration status
        if st.session_state.servers_config and st.session_state.migration_params:
            st.success("✅ Configuration complete! Go to Analysis section to run workload analysis.")
    
    elif page == "🚀 Analysis":
        st.markdown("## 🚀 Workload Analysis")
        
        # Check if configuration is complete
        if not st.session_state.servers_config:
            st.warning("⚠️ Please complete environment configuration first")
            if st.button("🔧 Go to Configuration"):
                st.session_state.main_navigation = "🔧 Configuration"
                st.rerun()
            return
        
        if not st.session_state.migration_params:
            st.warning("⚠️ Please complete migration parameters configuration first")
            if st.button("🔧 Go to Configuration"):
                st.session_state.main_navigation = "🔧 Configuration"
                st.rerun()
            return
        
        # Show current configuration summary
        st.markdown("### 📋 Configuration Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Environments:**")
            for env, specs in st.session_state.servers_config.items():
                st.markdown(f"• {env}: {specs['cpu']} vCPU, {specs['ram']} GB RAM")
        
        with col2:
            params = st.session_state.migration_params
            st.markdown("**Migration Details:**")
            st.markdown(f"• Data Size: {params['data_size_tb']} TB")
            st.markdown(f"• Timeline: {params['migration_timeline']} months")
            st.markdown(f"• PL/SQL Objects: {params['num_pl_sql_objects']:,}")
        
        with col3:
            st.markdown("**Cost Parameters:**")
            st.markdown(f"• Oracle License: ${params['oracle_license_cost']:,}/year")
            st.markdown(f"• Maintenance: ${params['manpower_cost']:,}/year")
            st.markdown(f"• Migration Budget: ${params['migration_budget']:,}")
        
        # Analysis button
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🔬 Run Analysis")
        
        analysis_col1, analysis_col2 = st.columns([2, 1])
        
        with analysis_col1:
            st.markdown("""
            **This analysis will provide:**
            - 💰 Detailed cost comparison between Oracle and AWS
            - 🏗️ Infrastructure recommendations for each environment
            - 📊 Migration complexity assessment
            - 🎯 Strategic migration approach recommendations
            - 📈 ROI projections and savings analysis
            """)
        
        with analysis_col2:
            if st.button("🚀 Analyze Workload", type="primary", use_container_width=True):
                results = analyze_workload(
                    st.session_state.servers_config,
                    st.session_state.migration_params
                )
                st.success("✅ Analysis complete! Check the Results section.")
                st.balloons()
        
        # Show previous results if available
        if st.session_state.analysis_results:
            st.markdown("### ✅ Previous Analysis Results")
            results = st.session_state.analysis_results
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Annual Savings", 
                         f"${results['cost_analysis']['total_annual_savings']:,.0f}")
            
            with summary_col2:
                st.metric("Migration Cost", 
                         f"${results['cost_analysis']['migration_costs']['total']:,.0f}")
            
            with summary_col3:
                st.metric("Complexity", 
                         f"{results['complexity_analysis']['score']}/100")
            
            st.info("📊 Go to Results section for detailed analysis")
    
    elif page == "📊 Results":
        st.markdown("## 📊 Analysis Results")
        display_results(st.session_state.analysis_results)

if __name__ == "__main__":
    main()