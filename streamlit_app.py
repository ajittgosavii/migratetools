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
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any

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

# MISSING FUNCTIONS TO ADD TO YOUR streamlit_app.py
# Add these functions BEFORE the main() function

def create_detailed_migration_timeline(params, complexity_score):
    """Create detailed migration timeline with phases and milestones"""
    
    timeline_months = params['migration_timeline']
    strategy = get_enhanced_migration_strategy(complexity_score, params)
    num_phases = strategy['recommended_phases']
    
    # Base phase templates
    base_phases = [
        {
            'name': 'Assessment & Planning',
            'duration_ratio': 0.15,
            'key_activities': [
                'Current state analysis',
                'Migration strategy finalization',
                'Team formation and training',
                'Tool selection and setup'
            ],
            'deliverables': [
                'Migration plan document',
                'Risk assessment',
                'Resource allocation plan',
                'Timeline and milestones'
            ]
        },
        {
            'name': 'Environment Setup',
            'duration_ratio': 0.20,
            'key_activities': [
                'AWS infrastructure provisioning',
                'MongoDB Atlas cluster setup',
                'Network configuration',
                'Security implementation'
            ],
            'deliverables': [
                'Development environment',
                'Testing environment',
                'Security baseline',
                'Monitoring setup'
            ]
        },
        {
            'name': 'Data Migration Design',
            'duration_ratio': 0.15,
            'key_activities': [
                'Schema design and mapping',
                'Data transformation rules',
                'Migration scripts development',
                'Validation procedures'
            ],
            'deliverables': [
                'Target schema design',
                'Data mapping document',
                'Migration scripts',
                'Validation framework'
            ]
        },
        {
            'name': 'Application Refactoring',
            'duration_ratio': 0.25,
            'key_activities': [
                'PL/SQL analysis and conversion',
                'Application layer updates',
                'Integration development',
                'Performance optimization'
            ],
            'deliverables': [
                'Refactored applications',
                'Updated integrations',
                'Performance benchmarks',
                'Code review reports'
            ]
        },
        {
            'name': 'Testing & Validation',
            'duration_ratio': 0.15,
            'key_activities': [
                'Unit and integration testing',
                'Performance testing',
                'User acceptance testing',
                'Data validation'
            ],
            'deliverables': [
                'Test execution reports',
                'Performance validation',
                'UAT sign-off',
                'Data integrity confirmation'
            ]
        },
        {
            'name': 'Go-Live & Optimization',
            'duration_ratio': 0.10,
            'key_activities': [
                'Production migration',
                'Cutover execution',
                'Post-migration monitoring',
                'Performance tuning'
            ],
            'deliverables': [
                'Production system',
                'Cutover report',
                'Monitoring dashboards',
                'Optimization recommendations'
            ]
        }
    ]
    
    # Select phases based on complexity
    selected_phases = base_phases[:num_phases]
    
    # Calculate phase durations
    timeline_phases = []
    start_date = datetime.now()
    
    for i, phase in enumerate(selected_phases):
        duration_weeks = int(timeline_months * 4.33 * phase['duration_ratio'])
        phase_start = start_date + timedelta(weeks=sum([p['duration_weeks'] for p in timeline_phases]))
        phase_end = phase_start + timedelta(weeks=duration_weeks)
        
        timeline_phases.append({
            'phase_number': i + 1,
            'name': phase['name'],
            'duration_weeks': duration_weeks,
            'duration_months': round(duration_weeks / 4.33, 1),
            'start_date': phase_start,
            'end_date': phase_end,
            'key_activities': phase['key_activities'],
            'deliverables': phase['deliverables'],
            'critical_path': i in [1, 3, 4]  # Environment setup, refactoring, testing are critical
        })
    
    return {
        'total_duration_months': timeline_months,
        'total_duration_weeks': sum([p['duration_weeks'] for p in timeline_phases]),
        'phases': timeline_phases,
        'critical_milestones': [
            {'name': 'Environment Ready', 'week': timeline_phases[1]['duration_weeks']},
            {'name': 'Data Migration Complete', 'week': sum([p['duration_weeks'] for p in timeline_phases[:3]])},
            {'name': 'Testing Complete', 'week': sum([p['duration_weeks'] for p in timeline_phases[:5]])},
            {'name': 'Go-Live', 'week': sum([p['duration_weeks'] for p in timeline_phases])}
        ]
    }

def get_enhanced_migration_strategy(complexity_score, params):
    """Get enhanced migration strategy with detailed recommendations"""
    
    if complexity_score < 25:
        strategy = {
            'name': 'Lift and Shift Plus',
            'approach': 'Direct migration with cloud optimization',
            'timeline': '3-5 months',
            'risk_level': 'Low',
            'effort_level': 'Low to Medium',
            'recommended_phases': 3,
            'key_benefits': [
                'Fastest time to market',
                'Minimal application changes',
                'Lower migration costs',
                'Quick ROI realization'
            ],
            'key_activities': [
                'Direct database migration using AWS DMS',
                'Minimal application refactoring',
                'Infrastructure optimization',
                'Performance tuning'
            ]
        }
    elif complexity_score < 50:
        strategy = {
            'name': 'Re-platform with Modernization',
            'approach': 'Migration with selective modernization',
            'timeline': '5-8 months',
            'risk_level': 'Medium',
            'effort_level': 'Medium',
            'recommended_phases': 4,
            'key_benefits': [
                'Balanced approach to modernization',
                'Improved performance and scalability',
                'Moderate timeline and cost',
                'Foundation for future enhancements'
            ],
            'key_activities': [
                'Selective PL/SQL refactoring',
                'Application layer modernization',
                'Database schema optimization',
                'Cloud-native service integration'
            ]
        }
    elif complexity_score < 75:
        strategy = {
            'name': 'Re-architect for Cloud',
            'approach': 'Comprehensive redesign for cloud-native architecture',
            'timeline': '8-14 months',
            'risk_level': 'Medium-High',
            'effort_level': 'High',
            'recommended_phases': 5,
            'key_benefits': [
                'Full cloud-native capabilities',
                'Maximum scalability and performance',
                'Long-term architectural benefits',
                'Modern development practices'
            ],
            'key_activities': [
                'Complete application redesign',
                'Microservices architecture adoption',
                'Advanced MongoDB features utilization',
                'DevOps and automation implementation'
            ]
        }
    else:
        strategy = {
            'name': 'Hybrid Transformation',
            'approach': 'Phased transformation with parallel systems',
            'timeline': '12-20 months',
            'risk_level': 'High',
            'effort_level': 'Very High',
            'recommended_phases': 6,
            'key_benefits': [
                'Minimized business disruption',
                'Gradual risk mitigation',
                'Comprehensive modernization',
                'Future-proof architecture'
            ],
            'key_activities': [
                'Parallel system development',
                'Gradual data migration',
                'Complete application transformation',
                'Advanced cloud services adoption'
            ]
        }
    
    return strategy

def calculate_risk_assessment(servers, params, complexity_analysis):
    """Calculate comprehensive risk assessment"""
    
    risks = []
    
    # Technical risks
    if complexity_analysis['factors']['PL/SQL Complexity'] > 60:
        risks.append({
            'category': 'Technical',
            'risk': 'PL/SQL Conversion Complexity',
            'probability': 'High' if complexity_analysis['factors']['PL_SQL_Complexity'] > 80 else 'Medium',
            'impact': 'High',
            'mitigation': 'Detailed code analysis, automated conversion tools, expert consultation',
            'owner': 'Technical Team Lead'
        })
    
    if params['data_size_tb'] > 50:
        risks.append({
            'category': 'Technical',
            'risk': 'Data Migration Performance',
            'probability': 'Medium',
            'impact': 'High',
            'mitigation': 'Parallel data streams, incremental migration, performance testing',
            'owner': 'Database Administrator'
        })
    
    # Timeline risks
    if params['migration_timeline'] < 6:
        risks.append({
            'category': 'Timeline',
            'risk': 'Aggressive Timeline',
            'probability': 'High',
            'impact': 'Medium',
            'mitigation': 'Resource augmentation, parallel work streams, scope prioritization',
            'owner': 'Project Manager'
        })
    
    # Business risks
    if len(servers) > 3:
        risks.append({
            'category': 'Business',
            'risk': 'Multi-Environment Coordination',
            'probability': 'Medium',
            'impact': 'Medium',
            'mitigation': 'Detailed coordination plan, environment-specific teams, clear communication',
            'owner': 'Program Manager'
        })
    
    if params['num_applications'] > 5:
        risks.append({
            'category': 'Business',
            'risk': 'Application Integration Complexity',
            'probability': 'Medium',
            'impact': 'High',
            'mitigation': 'Integration testing, API compatibility checks, rollback procedures',
            'owner': 'Integration Team Lead'
        })
    
    # Financial risks
    total_investment = 500000  # Placeholder for migration cost
    if total_investment > params['migration_budget']:
        risks.append({
            'category': 'Financial',
            'risk': 'Budget Overrun',
            'probability': 'Medium',
            'impact': 'High',
            'mitigation': 'Regular budget reviews, contingency planning, scope management',
            'owner': 'Financial Controller'
        })
    
    return {
        'total_risks': len(risks),
        'high_probability_risks': len([r for r in risks if r['probability'] == 'High']),
        'high_impact_risks': len([r for r in risks if r['impact'] == 'High']),
        'risk_details': risks,
        'overall_risk_level': calculate_overall_risk_level(risks)
    }

def calculate_overall_risk_level(risks):
    """Calculate overall risk level"""
    if not risks:
        return {'level': 'Low', 'color': '#38a169'}
    
    high_risks = len([r for r in risks if r['probability'] == 'High' and r['impact'] == 'High'])
    medium_risks = len([r for r in risks if (r['probability'] == 'High' and r['impact'] == 'Medium') or 
                                          (r['probability'] == 'Medium' and r['impact'] == 'High')])
    
    if high_risks > 2:
        return {'level': 'Very High', 'color': '#9f1239'}
    elif high_risks > 0 or medium_risks > 3:
        return {'level': 'High', 'color': '#e53e3e'}
    elif medium_risks > 0:
        return {'level': 'Medium', 'color': '#d69e2e'}
    else:
        return {'level': 'Low', 'color': '#38a169'}

# ALSO ADD THESE FUNCTIONS IF THEY'RE MISSING:

def create_migration_timeline_gantt(results):
    """Create detailed migration timeline Gantt chart"""
    st.markdown("## 📅 Migration Timeline & Gantt Chart")
    
    timeline = results['migration_timeline']
    
    # Prepare Gantt chart data
    gantt_data = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, phase in enumerate(timeline['phases']):
        gantt_data.append(dict(
            Task=phase['name'],
            Start=phase['start_date'],
            Finish=phase['end_date'],
            Resource=f"Phase {phase['phase_number']}",
            Description=f"{phase['duration_weeks']} weeks"
        ))
    
    # Create Gantt chart
    try:
        fig_gantt = ff.create_gantt(
            gantt_data,
            colors=colors[:len(timeline['phases'])],
            index_col='Resource',
            title='Migration Timeline - Gantt Chart',
            show_colorbar=True,
            bar_width=0.5,
            showgrid_x=True,
            showgrid_y=True
        )
        
        fig_gantt.update_layout(height=400)
        st.plotly_chart(fig_gantt, use_container_width=True)
    except Exception as e:
        st.error(f"Gantt chart creation failed: {e}")
        st.info("Displaying timeline as table instead:")
        
        # Fallback to table display
        timeline_data = []
        for phase in timeline['phases']:
            timeline_data.append({
                'Phase': f"{phase['phase_number']}. {phase['name']}",
                'Duration': f"{phase['duration_weeks']} weeks",
                'Start Date': phase['start_date'].strftime('%Y-%m-%d'),
                'End Date': phase['end_date'].strftime('%Y-%m-%d'),
                'Key Activities': ', '.join(phase['key_activities'][:3])
            })
        
        st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)
    
    # Timeline details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Phase Details")
        for phase in timeline['phases']:
            critical_indicator = "🔴" if phase.get('critical_path', False) else "🟢"
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-phase">{critical_indicator} Phase {phase['phase_number']}: {phase['name']}</div>
                <div class="timeline-duration">Duration: {phase['duration_weeks']} weeks ({phase['duration_months']} months)</div>
                <div class="timeline-tasks">
                    <strong>Key Activities:</strong><br>
                    {'<br>'.join(['• ' + activity for activity in phase['key_activities'][:3]])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Critical Milestones")
        for milestone in timeline['critical_milestones']:
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-phase">📌 {milestone['name']}</div>
                <div class="timeline-duration">Week {milestone['week']}</div>
            </div>
            """, unsafe_allow_html=True)

def create_waterfall_chart(results):
    """Create waterfall chart showing cost transformation"""
    st.markdown("## 💧 Cost Transformation Waterfall Chart")
    
    cost_analysis = results['cost_analysis']
    
    # Waterfall data
    categories = ['Current Oracle Cost', 'Oracle License Savings', 'Infrastructure Savings', 
                 'Maintenance Savings', 'AWS Infrastructure', 'AWS MongoDB', 'AWS Storage',
                 'AWS Security & Monitoring', 'Net Annual Savings']
    
    # Calculate values
    current_cost = cost_analysis['total_current_cost']
    aws_cost = cost_analysis['total_aws_cost']
    
    # Break down current costs (estimated)
    license_cost = current_cost * 0.4
    infrastructure_cost = current_cost * 0.35
    maintenance_cost = current_cost * 0.25
    
    # Break down AWS costs (estimated)
    aws_infrastructure = aws_cost * 0.4
    aws_mongodb = aws_cost * 0.35
    aws_storage = aws_cost * 0.15
    aws_security = aws_cost * 0.1
    
    values = [current_cost, -license_cost, -infrastructure_cost, -maintenance_cost,
             aws_infrastructure, aws_mongodb, aws_storage, aws_security, 
             cost_analysis['total_annual_savings']]
    
    # Create waterfall chart
    fig_waterfall = go.Figure()
    
    # Calculate cumulative values for positioning
    cumulative = [current_cost]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(cost_analysis['total_annual_savings'])
    
    # Add bars
    colors = ['blue', 'red', 'red', 'red', 'orange', 'orange', 'orange', 'orange', 'green']
    
    for i, (category, value, color) in enumerate(zip(categories, values, colors)):
        if i == 0:  # Starting value
            fig_waterfall.add_trace(go.Bar(
                x=[category], y=[value], name=category,
                marker_color=color, text=f'${value:,.0f}', textposition='auto'
            ))
        elif i == len(categories) - 1:  # Final value
            fig_waterfall.add_trace(go.Bar(
                x=[category], y=[value], name=category,
                marker_color=color, text=f'${value:,.0f}', textposition='auto'
            ))
        else:  # Intermediate values
            if value < 0:  # Savings
                fig_waterfall.add_trace(go.Bar(
                    x=[category], y=[abs(value)], base=[cumulative[i-1] + value],
                    name=category, marker_color=color,
                    text=f'${abs(value):,.0f}', textposition='auto'
                ))
            else:  # Costs
                fig_waterfall.add_trace(go.Bar(
                    x=[category], y=[value], base=[cumulative[i-1]],
                    name=category, marker_color=color,
                    text=f'${value:,.0f}', textposition='auto'
                ))
    
    fig_waterfall.update_layout(
        title='Annual Cost Transformation Waterfall',
        xaxis_title='Cost Components',
        yaxis_title='Annual Cost ($)',
        showlegend=False,
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # ROI projection over time
    st.markdown("### 📈 ROI Projection Over Time")
    
    years = list(range(1, 6))  # 5 year projection
    annual_savings = cost_analysis['total_annual_savings']
    migration_cost = cost_analysis['migration_costs']['total']
    
    cumulative_savings = [annual_savings * year for year in years]
    cumulative_investment = [migration_cost] * 5  # One-time investment
    net_benefit = [savings - migration_cost for savings in cumulative_savings]
    
    fig_roi = go.Figure()
    
    fig_roi.add_trace(go.Scatter(
        x=years, y=cumulative_savings, mode='lines+markers',
        name='Cumulative Savings', line=dict(color='green', width=3)
    ))
    
    fig_roi.add_trace(go.Scatter(
        x=years, y=cumulative_investment, mode='lines+markers',
        name='Migration Investment', line=dict(color='red', width=3, dash='dash')
    ))
    
    fig_roi.add_trace(go.Scatter(
        x=years, y=net_benefit, mode='lines+markers',
        name='Net Benefit', line=dict(color='blue', width=3),
        fill='tonexty', fillcolor='rgba(0,100,255,0.1)'
    ))
    
    fig_roi.update_layout(
        title='5-Year ROI Projection',
        xaxis_title='Years',
        yaxis_title='Amount ($)',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_roi, use_container_width=True)


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

def analyze_workload_enhanced(servers, params):
    """Enhanced workload analysis with comprehensive risk assessment and strategy recommendation"""
    
    with st.spinner('🔄 Performing comprehensive migration analysis...'):
        progress = st.progress(0)
        
        # Step 1: Instance Recommendations
        progress.progress(15)
        recommendations = get_instance_recommendations(servers)
        
        # Step 2: Cost Analysis
        progress.progress(30)
        cost_analysis = calculate_enhanced_cost_analysis(servers, params, recommendations)
        
        # Step 3: Complexity Analysis
        progress.progress(45)
        complexity_analysis = calculate_complexity_analysis(servers, params)
        
        # Step 4: Detailed Risk Assessment
        progress.progress(60)
        detailed_risk_assessment = calculate_detailed_risk_assessment(servers, params, complexity_analysis)
        
        # Step 5: Migration Strategy Analysis
        progress.progress(75)
        migration_strategy_analysis = determine_optimal_migration_strategy(
            servers, params, complexity_analysis, detailed_risk_assessment
        )
        
        # Step 6: Migration Timeline
        progress.progress(90)
        migration_timeline = create_detailed_migration_timeline(params, complexity_analysis['score'])
        
        # Step 7: Legacy Risk Assessment (for compatibility)
        progress.progress(95)
        risk_assessment = calculate_risk_assessment(servers, params, complexity_analysis)
        
        progress.progress(100)
        
        # Compile enhanced results
        results = {
            'servers': servers,
            'params': params,
            'recommendations': recommendations,
            'cost_analysis': cost_analysis,
            'complexity_analysis': complexity_analysis,
            'detailed_risk_assessment': detailed_risk_assessment,
            'migration_strategy_analysis': migration_strategy_analysis,
            'migration_strategy': migration_strategy_analysis['recommended_strategy'],  # For compatibility
            'migration_timeline': migration_timeline,
            'risk_assessment': risk_assessment,  # Legacy format for compatibility
            'timestamp': datetime.now()
        }
        
        st.session_state.analysis_results = results
        progress.empty()
    
    return results

# 5. REPLACE YOUR EXISTING display_enhanced_results FUNCTION WITH THIS UPDATED VERSION
def display_enhanced_results_updated(results):
    """Display enhanced results with all dashboards including new risk and strategy analysis"""
    if not results:
        st.info("👆 Please complete the analysis first")
        return
    
    st.markdown("## 📊 Enhanced Migration Analysis Results")
    
    # Navigation tabs for different dashboards
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "💰 Cost Summary", 
        "⚠️ Risk Assessment",
        "🎯 Migration Strategy",
        "🛡️ Risk Mitigation",
        "🔥 Heat Maps", 
        "📅 Timeline", 
        "💧 Waterfall", 
        "📄 Export"
    ])
    
    with tab1:
        create_cost_summary_dashboard(results)
    
    with tab2:
        create_risk_assessment_heatmap(results)
    
    with tab3:
        create_migration_strategy_dashboard(results)
    
    with tab4:
        create_risk_mitigation_dashboard(results)
    
    with tab5:
        create_environment_heatmap(results)
    
    with tab6:
        create_migration_timeline_gantt(results)
    
    with tab7:
        create_waterfall_chart(results)
    
    with tab8:
        st.markdown("## 📄 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Enhanced text report
            text_report = generate_enhanced_text_report(results)
            st.download_button(
                label="📄 Download Enhanced Report",
                data=text_report,
                file_name=f"Enhanced_Migration_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Risk assessment CSV
            if 'detailed_risk_assessment' in results:
                risk_data = prepare_risk_csv_export(results['detailed_risk_assessment'])
                st.download_button(
                    label="📊 Download Risk Assessment (CSV)",
                    data=risk_data,
                    file_name=f"Risk_Assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            # Strategy analysis CSV
            if 'migration_strategy_analysis' in results:
                strategy_data = prepare_strategy_csv_export(results['migration_strategy_analysis'])
                st.download_button(
                    label="📋 Download Strategy Analysis (CSV)",
                    data=strategy_data,
                    file_name=f"Strategy_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def generate_enhanced_text_report(results):
    """Generate enhanced comprehensive text report"""
    cost_analysis = results['cost_analysis']
    complexity_analysis = results['complexity_analysis']
    detailed_risk_assessment = results['detailed_risk_assessment']
    migration_strategy_analysis = results['migration_strategy_analysis']
    params = results['params']
    
    return f"""
COMPREHENSIVE ORACLE TO MONGODB MIGRATION ANALYSIS
=================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Current Annual Oracle Cost: ${cost_analysis['total_current_cost']:,.0f}
Projected Annual AWS Cost: ${cost_analysis['total_aws_cost']:,.0f}
Annual Savings: ${cost_analysis['total_annual_savings']:,.0f}
Migration Investment: ${cost_analysis['migration_costs']['total']:,.0f}
3-Year ROI: {cost_analysis['roi_3_year']:.1f}%
Payback Period: {cost_analysis['payback_period_months']:.1f} months
Complexity Score: {complexity_analysis['score']}/100
Overall Risk Level: {detailed_risk_assessment['risk_level']['level']}

RECOMMENDED MIGRATION STRATEGY
=============================
Strategy: {migration_strategy_analysis['recommended_strategy']['name']}
Approach: {migration_strategy_analysis['recommended_strategy']['description']}
Timeline: {migration_strategy_analysis['recommended_strategy']['timeline']}
Risk Level: {migration_strategy_analysis['recommended_strategy']['risk_level']}
Business Disruption: {migration_strategy_analysis['recommended_strategy']['business_disruption']}
Suitability Score: {migration_strategy_analysis['recommended_strategy']['suitability_score']:.1f}/100

RISK ASSESSMENT SUMMARY
=======================
Overall Risk Score: {detailed_risk_assessment['overall_risk_score']:.1f}/100
Risk Level: {detailed_risk_assessment['risk_level']['level']}
Required Action: {detailed_risk_assessment['risk_level']['action']}

Technical Risks:
- Database Complexity: {detailed_risk_assessment['risk_matrix']['Technical_Risks']['Database_Complexity']['score']:.0f}/100
- Application Integration: {detailed_risk_assessment['risk_matrix']['Technical_Risks']['Application_Integration']['score']:.0f}/100
- Performance Risk: {detailed_risk_assessment['risk_matrix']['Technical_Risks']['Performance_Risk']['score']:.0f}/100

Business Risks:
- Timeline Risk: {detailed_risk_assessment['risk_matrix']['Business_Risks']['Timeline_Risk']['score']:.0f}/100
- Budget Risk: {detailed_risk_assessment['risk_matrix']['Business_Risks']['Budget_Risk']['score']:.0f}/100
- Business Continuity: {detailed_risk_assessment['risk_matrix']['Business_Risks']['Business_Continuity']['score']:.0f}/100

STRATEGY ADVANTAGES
==================
{chr(10).join([f"• {pro}" for pro in migration_strategy_analysis['recommended_strategy']['pros']])}

CRITICAL SUCCESS FACTORS
========================
{chr(10).join([f"• {factor}" for factor in migration_strategy_analysis['success_factors']])}

IMPLEMENTATION ROADMAP
=====================
{chr(10).join([f"Phase: {phase['phase']} ({phase['duration']})" + chr(10) + chr(10).join([f"  • {activity}" for activity in phase['activities']]) for phase in migration_strategy_analysis['implementation_roadmap']])}

ENVIRONMENT ANALYSIS
===================
{chr(10).join([f"{env}: {specs['cpu']} vCPU, {specs['ram']} GB RAM, {specs['storage']} GB Storage" 
               for env, specs in results['servers'].items()])}

DETAILED COST BREAKDOWN
======================
Migration Team: ${cost_analysis['migration_costs']['migration_team']:,.0f}
Data Transfer: ${cost_analysis['migration_costs']['data_transfer']:,.0f}
AWS Services: ${cost_analysis['migration_costs']['aws_services']:,.0f}
Tools & Software: ${cost_analysis['migration_costs']['tools_software']:,.0f}
Training: ${cost_analysis['migration_costs']['training_certification']:,.0f}
Professional Services: ${cost_analysis['migration_costs']['professional_services']:,.0f}
Testing: ${cost_analysis['migration_costs']['testing_validation']:,.0f}
Contingency: ${cost_analysis['migration_costs']['contingency']:,.0f}

RECOMMENDATIONS
==============
1. Proceed with {migration_strategy_analysis['recommended_strategy']['name']} approach
2. Address high-risk areas identified in risk assessment
3. Allocate ${cost_analysis['migration_costs']['total']:,.0f} for migration
4. Plan for {results['migration_timeline']['total_duration_months']} month timeline
5. Implement comprehensive risk mitigation strategies
6. Focus on {complexity_analysis['complexity_level']['level'].lower()} complexity management
7. Ensure all prerequisites are met before starting migration
"""

def prepare_risk_csv_export(risk_assessment):
    """Prepare risk assessment data for CSV export"""
    risk_data = []
    
    for category, risks in risk_assessment['risk_matrix'].items():
        if category == 'category_score':
            continue
        for risk_name, risk_info in risks.items():
            if risk_name == 'category_score':
                continue
            risk_data.append({
                'Category': category.replace('_', ' '),
                'Risk_Type': risk_name.replace('_', ' '),
                'Score': risk_info['score'],
                'Level': risk_info['level']['level'],
                'Action_Required': risk_info['level']['action'],
                'Weight': risk_info['weight']
            })
    
    risk_df = pd.DataFrame(risk_data)
    return risk_df.to_csv(index=False)

def prepare_strategy_csv_export(strategy_analysis):
    """Prepare strategy analysis data for CSV export"""
    strategy_data = []
    
    for strategy_name, strategy_info in strategy_analysis['strategy_comparison'].items():
        strategy_data.append({
            'Strategy': strategy_info['name'],
            'Timeline': strategy_info['timeline'],
            'Risk_Level': strategy_info['risk_level'],
            'Cost_Efficiency': strategy_info['cost_efficiency'],
            'Business_Disruption': strategy_info['business_disruption'],
            'Suitability_Score': strategy_info['suitability_score'],
            'Best_For': strategy_info['best_for']
        })
    
    strategy_df = pd.DataFrame(strategy_data)
    return strategy_df.to_csv(index=False)

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

def calculate_enhanced_cost_analysis(servers, params, recommendations):
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
        if factors['PL/SQL Complexity'] > 60:  # Use 'factors' here
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
def calculate_detailed_risk_assessment(servers, params, complexity_analysis):
    """Calculate comprehensive risk assessment with detailed factors"""
    
    # Define risk categories and their factors
    risk_categories = {
        'Technical_Risks': {
            'Database_Complexity': {
                'weight': 0.25,
                'factors': ['pl_sql_objects', 'data_size', 'schema_complexity'],
                'calculation': lambda: min(100, (params['num_pl_sql_objects']/100 + params['data_size_tb']*2 + 30))
            },
            'Application_Integration': {
                'weight': 0.20,
                'factors': ['num_applications', 'api_complexity', 'data_dependencies'],
                'calculation': lambda: min(100, params['num_applications']*15 + 20)
            },
            'Performance_Risk': {
                'weight': 0.15,
                'factors': ['throughput_requirements', 'latency_sensitivity', 'concurrent_users'],
                'calculation': lambda: min(100, max([specs['throughput'] for specs in servers.values()])/500)
            },
            'Data_Migration_Risk': {
                'weight': 0.20,
                'factors': ['data_volume', 'downtime_tolerance', 'data_quality'],
                'calculation': lambda: min(100, params['data_size_tb']*3 + 25)
            },
            'Infrastructure_Risk': {
                'weight': 0.20,
                'factors': ['environment_count', 'resource_scale', 'network_complexity'],
                'calculation': lambda: min(100, len(servers)*12 + max([s['cpu']+s['ram']/4 for s in servers.values()]))
            }
        },
        'Business_Risks': {
            'Timeline_Risk': {
                'weight': 0.30,
                'factors': ['timeline_pressure', 'resource_availability', 'stakeholder_alignment'],
                'calculation': lambda: min(100, max(0, (12-params['migration_timeline'])*12))
            },
            'Budget_Risk': {
                'weight': 0.25,
                'factors': ['cost_overrun_probability', 'contingency_buffer', 'financial_constraints'],
                'calculation': lambda: min(100, max(0, 50 - (params['migration_budget']/100000)*2))
            },
            'Business_Continuity': {
                'weight': 0.25,
                'factors': ['service_availability', 'user_impact', 'revenue_impact'],
                'calculation': lambda: min(100, len(servers)*8 + params['num_applications']*6)
            },
            'Change_Management': {
                'weight': 0.20,
                'factors': ['team_readiness', 'training_needs', 'cultural_adaptation'],
                'calculation': lambda: min(100, 40 + params['num_applications']*3)
            }
        },
        'Security_Risks': {
            'Data_Security': {
                'weight': 0.35,
                'factors': ['sensitive_data', 'compliance_requirements', 'encryption_needs'],
                'calculation': lambda: min(100, params['data_size_tb']*2 + 45)
            },
            'Access_Control': {
                'weight': 0.25,
                'factors': ['user_management', 'privilege_escalation', 'authentication'],
                'calculation': lambda: min(100, params['num_applications']*8 + 30)
            },
            'Network_Security': {
                'weight': 0.25,
                'factors': ['firewall_config', 'vpc_setup', 'encryption_transit'],
                'calculation': lambda: min(100, len(servers)*10 + 35)
            },
            'Compliance_Risk': {
                'weight': 0.15,
                'factors': ['regulatory_requirements', 'audit_trail', 'data_governance'],
                'calculation': lambda: min(100, 50 + params['data_size_tb'])
            }
        },
        'Operational_Risks': {
            'Monitoring_Observability': {
                'weight': 0.30,
                'factors': ['monitoring_gaps', 'alerting_setup', 'performance_tracking'],
                'calculation': lambda: min(100, complexity_analysis['score']*0.6)
            },
            'Backup_Recovery': {
                'weight': 0.25,
                'factors': ['backup_strategy', 'rto_requirements', 'disaster_recovery'],
                'calculation': lambda: min(100, params['data_size_tb']*1.5 + 30)
            },
            'Maintenance_Support': {
                'weight': 0.25,
                'factors': ['skill_availability', 'vendor_support', 'documentation'],
                'calculation': lambda: min(100, 45 + len(servers)*5)
            },
            'Scaling_Capacity': {
                'weight': 0.20,
                'factors': ['growth_projection', 'elasticity_needs', 'capacity_planning'],
                'calculation': lambda: min(100, max([s['cpu'] for s in servers.values()])*1.5)
            }
        }
    }
    
    # Calculate risk scores
    risk_matrix = {}
    environment_risks = {}
    
    for category, risks in risk_categories.items():
        category_scores = {}
        for risk_name, risk_config in risks.items():
            risk_score = risk_config['calculation']()
            category_scores[risk_name] = {
                'score': risk_score,
                'weight': risk_config['weight'],
                'factors': risk_config['factors'],
                'level': get_risk_level(risk_score)
            }
        
        risk_matrix[category] = category_scores
        
        # Calculate weighted category score
        weighted_score = sum(risk['score'] * risk['weight'] for risk in category_scores.values())
        risk_matrix[category]['category_score'] = weighted_score
    
    # Calculate environment-specific risks
    for env_name, specs in servers.items():
        env_risks = calculate_environment_specific_risks(env_name, specs, params)
        environment_risks[env_name] = env_risks
    
    # Calculate overall risk score
    overall_risk_score = sum([
        risk_matrix['Technical_Risks']['category_score'] * 0.35,
        risk_matrix['Business_Risks']['category_score'] * 0.25,
        risk_matrix['Security_Risks']['category_score'] * 0.25,
        risk_matrix['Operational_Risks']['category_score'] * 0.15
    ])
    
    return {
        'risk_matrix': risk_matrix,
        'environment_risks': environment_risks,
        'overall_risk_score': overall_risk_score,
        'risk_level': get_risk_level(overall_risk_score),
        'mitigation_strategies': generate_mitigation_strategies(risk_matrix),
        'risk_timeline': generate_risk_timeline(risk_matrix, params)
    }

def calculate_environment_specific_risks(env_name, specs, params):
    """Calculate risks specific to each environment"""
    
    # Determine environment type
    env_type = categorize_environment(env_name)
    
    base_risks = {
        'Resource_Adequacy': min(100, (specs['cpu'] * 2 + specs['ram'] * 0.5) * 0.8),
        'Performance_Risk': min(100, specs['throughput'] / 300),
        'Storage_Risk': min(100, specs['storage'] / 100),
        'Availability_Risk': get_availability_risk(env_type),
        'Complexity_Risk': get_environment_complexity_risk(env_type, specs)
    }
    
    return base_risks

def categorize_environment(env_name):
    """Categorize environment type"""
    env_lower = env_name.lower()
    if any(term in env_lower for term in ['prod', 'production', 'live']):
        return 'Production'
    elif any(term in env_lower for term in ['stage', 'staging', 'preprod']):
        return 'Staging'
    elif any(term in env_lower for term in ['test', 'testing', 'qa', 'uat']):
        return 'Testing'
    elif any(term in env_lower for term in ['dev', 'development', 'sandbox']):
        return 'Development'
    else:
        return 'Other'

def get_availability_risk(env_type):
    """Get availability risk based on environment type"""
    risk_mapping = {
        'Production': 85,
        'Staging': 60,
        'Testing': 40,
        'Development': 25,
        'Other': 50
    }
    return risk_mapping.get(env_type, 50)

def get_environment_complexity_risk(env_type, specs):
    """Calculate environment complexity risk"""
    base_complexity = {
        'Production': 70,
        'Staging': 50,
        'Testing': 35,
        'Development': 25,
        'Other': 45
    }
    
    # Adjust based on resource scale
    resource_factor = (specs['cpu'] + specs['ram']/4) / 50
    complexity = base_complexity.get(env_type, 45) * min(2.0, resource_factor)
    
    return min(100, complexity)

def get_risk_level(score):
    """Get risk level description based on score"""
    if score < 25:
        return {'level': 'Low', 'color': '#38a169', 'action': 'Monitor'}
    elif score < 50:
        return {'level': 'Medium', 'color': '#d69e2e', 'action': 'Mitigate'}
    elif score < 75:
        return {'level': 'High', 'color': '#e53e3e', 'action': 'Urgent Action'}
    else:
        return {'level': 'Critical', 'color': '#9f1239', 'action': 'Immediate Response'}

def generate_mitigation_strategies(risk_matrix):
    """Generate specific mitigation strategies based on risk analysis"""
    
    strategies = {}
    
    for category, risks in risk_matrix.items():
        if category == 'category_score':
            continue
            
        category_strategies = []
        
        for risk_name, risk_data in risks.items():
            if risk_name == 'category_score':
                continue
                
            if risk_data['score'] > 50:  # High risk
                strategy = get_mitigation_strategy(category, risk_name, risk_data['score'])
                if strategy:
                    category_strategies.append(strategy)
        
        if category_strategies:
            strategies[category] = category_strategies
    
    return strategies

def get_mitigation_strategy(category, risk_name, score):
    """Get specific mitigation strategy for a risk"""
    
    strategies = {
        'Technical_Risks': {
            'Database_Complexity': {
                'strategy': 'Implement automated PL/SQL conversion tools and conduct thorough code analysis',
                'timeline': '2-4 weeks',
                'resources': ['Database Architect', 'Migration Specialist'],
                'cost_impact': 'Medium'
            },
            'Application_Integration': {
                'strategy': 'Develop comprehensive API testing framework and integration validation',
                'timeline': '3-6 weeks',
                'resources': ['Integration Specialist', 'QA Engineer'],
                'cost_impact': 'Medium'
            },
            'Performance_Risk': {
                'strategy': 'Conduct performance benchmarking and implement optimization strategies',
                'timeline': '2-3 weeks',
                'resources': ['Performance Engineer', 'Database Tuning Specialist'],
                'cost_impact': 'Low'
            }
        },
        'Business_Risks': {
            'Timeline_Risk': {
                'strategy': 'Implement parallel development streams and increase resource allocation',
                'timeline': 'Immediate',
                'resources': ['Project Manager', 'Additional Team Members'],
                'cost_impact': 'High'
            },
            'Budget_Risk': {
                'strategy': 'Implement strict budget monitoring and scope management',
                'timeline': 'Ongoing',
                'resources': ['Financial Controller', 'Project Manager'],
                'cost_impact': 'Low'
            }
        },
        'Security_Risks': {
            'Data_Security': {
                'strategy': 'Implement end-to-end encryption and data classification framework',
                'timeline': '2-4 weeks',
                'resources': ['Security Architect', 'Data Protection Officer'],
                'cost_impact': 'Medium'
            }
        },
        'Operational_Risks': {
            'Monitoring_Observability': {
                'strategy': 'Deploy comprehensive monitoring and alerting infrastructure',
                'timeline': '1-2 weeks',
                'resources': ['DevOps Engineer', 'Monitoring Specialist'],
                'cost_impact': 'Low'
            }
        }
    }
    
    return strategies.get(category, {}).get(risk_name)

def generate_risk_timeline(risk_matrix, params):
    """Generate risk timeline showing when risks peak during migration"""
    
    migration_months = params['migration_timeline']
    timeline_risks = []
    
    # Map risks to migration phases
    phase_risk_mapping = {
        1: ['Timeline_Risk', 'Budget_Risk'],  # Planning phase
        2: ['Infrastructure_Risk', 'Security_Risk'],  # Setup phase
        3: ['Data_Migration_Risk', 'Performance_Risk'],  # Migration phase
        4: ['Application_Integration', 'Testing_Risk'],  # Testing phase
        5: ['Business_Continuity', 'Change_Management']  # Go-live phase
    }
    
    months_per_phase = migration_months / 5
    
    for phase, risk_types in phase_risk_mapping.items():
        phase_start = (phase - 1) * months_per_phase
        phase_end = phase * months_per_phase
        
        timeline_risks.append({
            'phase': phase,
            'start_month': phase_start,
            'end_month': phase_end,
            'peak_risks': risk_types,
            'mitigation_window': f"Months {phase_start:.1f}-{phase_end:.1f}"
        })
    
    return timeline_risks

def determine_optimal_migration_strategy(servers, params, complexity_analysis, risk_assessment):
    """Determine optimal migration strategy based on comprehensive analysis"""
    
    # Calculate strategy factors
    factors = {
        'complexity_score': complexity_analysis['score'],
        'risk_score': risk_assessment['overall_risk_score'],
        'data_volume': params['data_size_tb'],
        'timeline_pressure': max(0, 12 - params['migration_timeline']),
        'budget_constraints': params['migration_budget'] / 1000000,  # Convert to millions
        'environment_count': len(servers),
        'application_complexity': params['num_applications'] * params['num_pl_sql_objects'] / 1000
    }
    
    # Strategy scoring matrix
    strategies = {
        'Big_Bang_Migration': {
            'name': 'Big Bang Migration',
            'description': 'Complete migration in a single cutover window',
            'best_for': 'Small to medium databases with minimal complexity',
            'timeline': '3-6 months',
            'risk_level': 'High',
            'cost_efficiency': 'High',
            'business_disruption': 'High',
            'suitability_score': calculate_big_bang_score(factors),
            'pros': [
                'Fastest implementation',
                'Lower overall cost',
                'Simpler project management',
                'Quick ROI realization'
            ],
            'cons': [
                'Higher risk of extended downtime',
                'Limited rollback options',
                'Requires extensive testing',
                'High pressure on execution'
            ],
            'prerequisites': [
                'Comprehensive testing environment',
                'Detailed rollback plan',
                'Extended maintenance window availability',
                'Experienced migration team'
            ]
        },
        'Phased_Migration': {
            'name': 'Phased Migration',
            'description': 'Migrate applications/modules in planned phases',
            'best_for': 'Medium to large environments with multiple applications',
            'timeline': '6-12 months',
            'risk_level': 'Medium',
            'cost_efficiency': 'Medium',
            'business_disruption': 'Medium',
            'suitability_score': calculate_phased_score(factors),
            'pros': [
                'Reduced risk per phase',
                'Learning from early phases',
                'Gradual user adaptation',
                'Better change management'
            ],
            'cons': [
                'Longer overall timeline',
                'Complex data synchronization',
                'Higher coordination overhead',
                'Potential integration challenges'
            ],
            'prerequisites': [
                'Application dependency mapping',
                'Data synchronization strategy',
                'Phase-specific testing',
                'Cross-phase coordination plan'
            ]
        },
        'Parallel_Run': {
            'name': 'Parallel Run Migration',
            'description': 'Run both systems in parallel before final cutover',
            'best_for': 'Mission-critical systems requiring zero downtime',
            'timeline': '8-15 months',
            'risk_level': 'Low',
            'cost_efficiency': 'Low',
            'business_disruption': 'Low',
            'suitability_score': calculate_parallel_score(factors),
            'pros': [
                'Minimal business risk',
                'Extensive validation period',
                'Gradual transition capability',
                'Strong rollback options'
            ],
            'cons': [
                'Highest cost approach',
                'Complex data synchronization',
                'Extended timeline',
                'Resource intensive'
            ],
            'prerequisites': [
                'Real-time data synchronization',
                'Dual infrastructure capacity',
                'Comprehensive monitoring',
                'Extended budget allocation'
            ]
        },
        'Hybrid_Approach': {
            'name': 'Hybrid Approach',
            'description': 'Combination of strategies based on application criticality',
            'best_for': 'Complex environments with varying application criticalities',
            'timeline': '9-18 months',
            'risk_level': 'Medium-Low',
            'cost_efficiency': 'Medium',
            'business_disruption': 'Low-Medium',
            'suitability_score': calculate_hybrid_score(factors),
            'pros': [
                'Optimized risk management',
                'Tailored approach per application',
                'Balanced cost and risk',
                'Flexible execution'
            ],
            'cons': [
                'Complex project management',
                'Multiple migration patterns',
                'Coordination challenges',
                'Varied skill requirements'
            ],
            'prerequisites': [
                'Application criticality assessment',
                'Multi-strategy planning',
                'Diverse skill sets',
                'Flexible resource allocation'
            ]
        }
    }
    
    # Select optimal strategy
    best_strategy = max(strategies.items(), key=lambda x: x[1]['suitability_score'])
    
    return {
        'recommended_strategy': best_strategy[1],
        'strategy_comparison': strategies,
        'decision_factors': factors,
        'implementation_roadmap': generate_implementation_roadmap(best_strategy[1], factors),
        'success_factors': get_success_factors(best_strategy[1])
    }

def calculate_big_bang_score(factors):
    """Calculate suitability score for Big Bang migration"""
    score = 100
    
    # Penalize based on complexity and risk
    score -= factors['complexity_score'] * 0.8
    score -= factors['risk_score'] * 0.6
    score -= factors['data_volume'] * 2
    score -= factors['environment_count'] * 5
    score -= factors['application_complexity'] * 3
    
    # Bonus for simple scenarios
    if factors['data_volume'] < 10 and factors['environment_count'] <= 2:
        score += 20
    
    return max(0, score)

def calculate_phased_score(factors):
    """Calculate suitability score for Phased migration"""
    score = 80
    
    # Optimal for medium complexity
    if 30 <= factors['complexity_score'] <= 70:
        score += 15
    
    # Good for multiple environments
    if factors['environment_count'] > 2:
        score += 10
    
    # Penalize for very high complexity
    if factors['complexity_score'] > 80:
        score -= 20
    
    return max(0, score)

def calculate_parallel_score(factors):
    """Calculate suitability score for Parallel migration"""
    score = 60
    
    # Best for high-risk, high-complexity scenarios
    if factors['risk_score'] > 60:
        score += 25
    
    if factors['complexity_score'] > 70:
        score += 20
    
    # Requires adequate budget
    if factors['budget_constraints'] < 0.5:  # Less than 500K
        score -= 30
    
    return max(0, score)

def calculate_hybrid_score(factors):
    """Calculate suitability score for Hybrid migration"""
    score = 70
    
    # Good for complex, multi-environment scenarios
    if factors['environment_count'] > 3:
        score += 15
    
    if factors['application_complexity'] > 5:
        score += 10
    
    # Balanced approach bonus
    if 40 <= factors['complexity_score'] <= 80:
        score += 10
    
    return max(0, score)

def generate_implementation_roadmap(strategy, factors):
    """Generate detailed implementation roadmap for selected strategy"""
    
    roadmap_templates = {
        'Big Bang Migration': [
            {
                'phase': 'Preparation',
                'duration': '4-6 weeks',
                'activities': [
                    'Complete environment analysis',
                    'Develop migration scripts',
                    'Setup target infrastructure',
                    'Create rollback procedures'
                ]
            },
            {
                'phase': 'Testing',
                'duration': '2-3 weeks',
                'activities': [
                    'Execute test migrations',
                    'Performance validation',
                    'User acceptance testing',
                    'Rollback testing'
                ]
            },
            {
                'phase': 'Go-Live',
                'duration': '1-2 weeks',
                'activities': [
                    'Final data migration',
                    'Application cutover',
                    'Smoke testing',
                    'Post-migration monitoring'
                ]
            }
        ],
        'Phased Migration': [
            {
                'phase': 'Phase Planning',
                'duration': '3-4 weeks',
                'activities': [
                    'Application dependency mapping',
                    'Phase sequence design',
                    'Data synchronization planning',
                    'Risk assessment per phase'
                ]
            },
            {
                'phase': 'Phase Execution',
                'duration': '16-20 weeks',
                'activities': [
                    'Execute migration phases',
                    'Inter-phase validation',
                    'Progressive rollout',
                    'Continuous monitoring'
                ]
            },
            {
                'phase': 'Consolidation',
                'duration': '2-3 weeks',
                'activities': [
                    'Final phase completion',
                    'System integration testing',
                    'Performance optimization',
                    'Documentation update'
                ]
            }
        ]
        # Add other strategy roadmaps as needed
    }
    
    return roadmap_templates.get(strategy['name'], [])

def get_success_factors(strategy):
    """Get critical success factors for the strategy"""
    
    success_factors = {
        'Big Bang Migration': [
            'Comprehensive testing in production-like environment',
            'Detailed rollback plan and procedures',
            'Experienced migration team availability',
            'Adequate maintenance window',
            'Strong project governance',
            'Clear communication plan'
        ],
        'Phased Migration': [
            'Accurate application dependency mapping',
            'Robust data synchronization mechanism',
            'Strong phase coordination',
            'Continuous monitoring and feedback',
            'Flexible resource allocation',
            'Change management excellence'
        ],
        'Parallel Run Migration': [
            'Real-time data synchronization',
            'Comprehensive monitoring setup',
            'Adequate infrastructure capacity',
            'Extended budget commitment',
            'Strong validation processes',
            'Risk management excellence'
        ],
        'Hybrid Approach': [
            'Application criticality assessment',
            'Multi-strategy expertise',
            'Complex project coordination',
            'Flexible execution capability',
            'Diverse skill set availability',
            'Adaptive planning approach'
        ]
    }
    
    return success_factors.get(strategy['name'], [])

def create_risk_assessment_heatmap(results):
    """Create comprehensive risk assessment heatmap"""
    st.markdown("## ⚠️ Risk Assessment Heat Map")
    
    risk_assessment = results['detailed_risk_assessment']
    risk_matrix = risk_assessment['risk_matrix']
    environment_risks = risk_assessment['environment_risks']
    
    # Create risk category heatmap
    st.markdown("### 🎯 Risk Categories Analysis")
    
    categories = list(risk_matrix.keys())
    risk_types = []
    risk_scores = []
    
    for category in categories:
        for risk_name, risk_data in risk_matrix[category].items():
            if risk_name != 'category_score':
                risk_types.append(f"{category.replace('_', ' ')}")
                risk_scores.append([risk_data['score']])
    
    # Create main risk heatmap
    fig_risk = go.Figure(data=go.Heatmap(
        z=risk_scores,
        x=['Risk Score'],
        y=risk_types,
        colorscale=[
            [0, '#38a169'],      # Green for low risk
            [0.25, '#d69e2e'],   # Yellow for medium risk
            [0.5, '#e53e3e'],    # Red for high risk
            [1, '#9f1239']       # Dark red for critical risk
        ],
        text=[[f'{score[0]:.0f}' for score in risk_scores]],
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hoverongaps=False,
        colorbar=dict(
            title="Risk Level",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["Low", "Medium", "High", "Critical", "Extreme"]
        )
    ))
    
    fig_risk.update_layout(
        title='Migration Risk Assessment by Category',
        height=600,
        yaxis=dict(autorange="reversed")
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Environment-specific risk heatmap
    st.markdown("### 🏢 Environment-Specific Risk Analysis")
    
    if environment_risks:
        env_names = list(environment_risks.keys())
        risk_categories = list(next(iter(environment_risks.values())).keys())
        
        env_risk_matrix = []
        for env in env_names:
            env_scores = [environment_risks[env][category] for category in risk_categories]
            env_risk_matrix.append(env_scores)
        
        fig_env_risk = go.Figure(data=go.Heatmap(
            z=env_risk_matrix,
            x=[cat.replace('_', ' ') for cat in risk_categories],
            y=env_names,
            colorscale='Reds',
            text=[[f'{val:.0f}' for val in row] for row in env_risk_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_env_risk.update_layout(
            title='Environment-Specific Risk Distribution',
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_env_risk, use_container_width=True)

def create_migration_strategy_dashboard(results):
    """Create comprehensive migration strategy dashboard"""
    st.markdown("## 🎯 Optimal Migration Strategy Dashboard")
    
    strategy_analysis = results['migration_strategy_analysis']
    recommended = strategy_analysis['recommended_strategy']
    comparison = strategy_analysis['strategy_comparison']
    
    # Strategy recommendation card
    st.markdown(f"""
    <div class="strategy-card strategy-{recommended['risk_level'].lower().replace('-', '_')}-risk">
        <h2>🏆 Recommended Strategy: {recommended['name']}</h2>
        <p><strong>Description:</strong> {recommended['description']}</p>
        <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
            <div><strong>Timeline:</strong> {recommended['timeline']}</div>
            <div><strong>Risk Level:</strong> {recommended['risk_level']}</div>
            <div><strong>Cost Efficiency:</strong> {recommended['cost_efficiency']}</div>
        </div>
        <p><strong>Best For:</strong> {recommended['best_for']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy comparison matrix
    st.markdown("### 📊 Strategy Comparison Matrix")
    
    comparison_data = []
    for strategy_name, strategy_data in comparison.items():
        comparison_data.append({
            'Strategy': strategy_data['name'],
            'Timeline': strategy_data['timeline'],
            'Risk Level': strategy_data['risk_level'],
            'Cost Efficiency': strategy_data['cost_efficiency'],
            'Business Disruption': strategy_data['business_disruption'],
            'Suitability Score': f"{strategy_data['suitability_score']:.1f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Pros and cons analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Advantages")
        for pro in recommended['pros']:
            st.markdown(f"• {pro}")
    
    with col2:
        st.markdown("### ⚠️ Considerations")
        for con in recommended['cons']:
            st.markdown(f"• {con}")

def create_risk_mitigation_dashboard(results):
    """Create risk mitigation strategies dashboard"""
    st.markdown("## 🛡️ Risk Mitigation Strategies")
    
    risk_assessment = results['detailed_risk_assessment']
    mitigation_strategies = risk_assessment['mitigation_strategies']
    
    # Overall risk level indicator
    overall_risk = risk_assessment['risk_level']
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {overall_risk['color']};">
        <div class="metric-value" style="color: {overall_risk['color']};">{overall_risk['level']}</div>
        <div class="metric-label">Overall Risk Level</div>
        <div class="metric-change">Action Required: {overall_risk['action']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mitigation strategies by category
    for category, strategies in mitigation_strategies.items():
        if strategies:  # Only show categories with strategies
            st.markdown(f"### 🎯 {category.replace('_', ' ')} Mitigation")
            
            for i, strategy in enumerate(strategies):
                if strategy:  # Check if strategy exists
                    with st.expander(f"Strategy {i+1}: {strategy.get('strategy', 'Risk Mitigation')}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Timeline:** {strategy.get('timeline', 'TBD')}")
                            st.markdown(f"**Cost Impact:** {strategy.get('cost_impact', 'Medium')}")
                        
                        with col2:
                            st.markdown("**Required Resources:**")
                            resources = strategy.get('resources', ['TBD'])
                            for resource in resources:
                                st.markdown(f"• {resource}")

def main():
    """Enhanced main application function with comprehensive risk assessment"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="enterprise-header">
        <h1>🏢 Enterprise Oracle to MongoDB Migration Analyzer</h1>
        <p>Advanced analytics with comprehensive risk assessment and strategic planning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Select Section:",
            ["🔧 Configuration", "🚀 Analysis", "📊 Enhanced Results"],
            key="main_navigation"
        )
        
        # Enhanced status indicators
        st.markdown("### 📋 Analysis Status")
        if st.session_state.servers_config:
            st.success(f"✅ {len(st.session_state.servers_config)} environments configured")
        else:
            st.warning("⚠️ Configure environments")
        
        if st.session_state.migration_params:
            st.success("✅ Migration parameters set")
        else:
            st.warning("⚠️ Set migration parameters")
        
        if st.session_state.analysis_results:
            st.success("✅ Comprehensive analysis complete")
            # Enhanced quick metrics
            results = st.session_state.analysis_results
            
            # Cost metrics
            st.metric("Annual Savings", f"${results['cost_analysis']['total_annual_savings']:,.0f}")
            st.metric("ROI", f"{results['cost_analysis']['roi_3_year']:.1f}%")
            
            # Risk metrics
            if 'detailed_risk_assessment' in results:
                risk_level = results['detailed_risk_assessment']['risk_level']
                st.metric("Risk Level", risk_level['level'], delta=risk_level['action'])
            
            # Strategy recommendation
            if 'migration_strategy_analysis' in results:
                strategy = results['migration_strategy_analysis']['recommended_strategy']
                st.info(f"💡 Recommended: {strategy['name']}")
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
            st.success("✅ Configuration complete! Go to Analysis section to run comprehensive workload analysis.")
    
    elif page == "🚀 Analysis":
        st.markdown("## 🚀 Comprehensive Migration Analysis")
        
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
        
        # Enhanced analysis description
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🔬 Run Comprehensive Migration Analysis")
        
        analysis_col1, analysis_col2 = st.columns([2, 1])
        
        with analysis_col1:
            st.markdown("""
            **This comprehensive analysis includes:**
            - 💰 **Cost Analysis:** Detailed cost breakdowns and ROI calculations
            - ⚠️ **Risk Assessment:** Multi-dimensional risk analysis with heat maps
            - 🎯 **Strategy Recommendation:** AI-powered optimal migration strategy selection
            - 🛡️ **Risk Mitigation:** Actionable mitigation strategies for identified risks
            - 🔥 **Environment Analysis:** Resource and complexity heat maps
            - 📅 **Timeline Planning:** Detailed Gantt charts and critical path analysis
            - 💧 **Cost Transformation:** Waterfall analysis of cost changes
            - 📄 **Professional Reports:** Comprehensive PDF and CSV exports
            """)
        
        with analysis_col2:
            if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
                results = analyze_workload_enhanced(  # <-- CHANGED FROM analyze_workload
                    st.session_state.servers_config,
                    st.session_state.migration_params
                )
                st.success("✅ Comprehensive analysis complete! Check the Enhanced Results section.")
                st.balloons()
        
        # Show previous results if available
        if st.session_state.analysis_results:
            st.markdown("### ✅ Previous Analysis Results")
            results = st.session_state.analysis_results
            
            # Enhanced summary metrics
            summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
            
            with summary_col1:
                st.metric("Annual Savings", 
                         f"${results['cost_analysis']['total_annual_savings']:,.0f}")
            
            with summary_col2:
                st.metric("Migration Cost", 
                         f"${results['cost_analysis']['migration_costs']['total']:,.0f}")
            
            with summary_col3:
                st.metric("3-Year ROI", 
                         f"{results['cost_analysis']['roi_3_year']:.1f}%")
            
            with summary_col4:
                if 'detailed_risk_assessment' in results:
                    risk_level = results['detailed_risk_assessment']['risk_level']
                    st.metric("Risk Level", risk_level['level'])
                else:
                    st.metric("Complexity", f"{results['complexity_analysis']['score']}/100")
            
            with summary_col5:
                if 'migration_strategy_analysis' in results:
                    strategy = results['migration_strategy_analysis']['recommended_strategy']
                    st.metric("Strategy", strategy['name'][:15] + "...")
                else:
                    st.metric("Timeline", f"{results['migration_timeline']['total_duration_months']} months")
            
            st.info("📊 Go to Enhanced Results section for detailed dashboards and comprehensive analysis")
    
    elif page == "📊 Enhanced Results":
        st.markdown("## 📊 Comprehensive Migration Analysis Results")
        display_enhanced_results_updated(st.session_state.analysis_results)  # <-- CHANGED FROM display_enhanced_results
        
        
if __name__ == "__main__":
    main()
    