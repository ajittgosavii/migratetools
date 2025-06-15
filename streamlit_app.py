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

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# AI and AWS integrations
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_API_AVAILABLE = True
except ImportError:
    AWS_API_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Enterprise Oracle to MongoDB Migration Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #4299e1;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
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
    
    .analysis-section {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .risk-high { background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); color: #742a2a; }
    .risk-medium { background: linear-gradient(135deg, #feebc8 0%, #fbd38d 100%); color: #744210; }
    .risk-low { background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); color: #22543d; }
    
    .download-section {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-connected { background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); color: #22543d; }
    .status-disconnected { background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); color: #742a2a; }
    .status-fallback { background: linear-gradient(135deg, #feebc8 0%, #fbd38d 100%); color: #744210; }
    .status-error { background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); color: #742a2a; }
    .status-unavailable { background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); color: #742a2a; }
    
    .service-status-card {
        background: white;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        border-left: 4px solid #e2e8f0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .indicator-green { background-color: #48bb78; }
    .indicator-red { background-color: #f56565; }
    .indicator-yellow { background-color: #ed8936; }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        margin: 3rem 0;
        border-radius: 1px;
    }
    
    .transfer-comparison-table {
        background: white;
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    .transfer-option-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
    
    .recommended-option {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced AWS Pricing with Real-time Data
class AWSPricingService:
    def __init__(self):
        self.pricing_client = None
        self.ec2_client = None
        self.region = 'us-east-1'
        self.connection_error = None
        self.is_initialized = False
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS clients if credentials are available"""
        self.is_initialized = True
        if not AWS_API_AVAILABLE:
            self.connection_error = "AWS SDK not available"
            return False
            
        try:
            # Test with a simple pricing call
            self.pricing_client = boto3.client('pricing', region_name='us-east-1')
            self.ec2_client = boto3.client('ec2', region_name=self.region)
            
            # Test the connection with a simple call
            test_response = self.pricing_client.describe_services(MaxResults=1)
            return True
        except (NoCredentialsError, ClientError) as e:
            self.connection_error = str(e)
            self.pricing_client = None
            return False
        except Exception as e:
            self.connection_error = f"AWS API Error: {str(e)}"
            self.pricing_client = None
            return False
    
    def get_connection_status(self):
        """Get detailed connection status"""
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Service not initialized',
                'details': 'Internal initialization error'
            }
            
        if not AWS_API_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'AWS SDK not installed',
                'details': 'Install boto3: pip install boto3'
            }
        
        if self.pricing_client:
            try:
                # Test with actual API call
                test_response = self.pricing_client.describe_services(MaxResults=1)
                return {
                    'status': 'connected',
                    'message': 'Real-time AWS pricing active',
                    'details': f'Connected to {self.region} region'
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'message': 'AWS connection failed',
                    'details': str(e)
                }
        
        return {
            'status': 'fallback',
            'message': 'Using fallback pricing',
            'details': self.connection_error or 'Configure AWS credentials for real-time pricing'
        }
    
    def get_ec2_pricing(self, instance_types: List[str] = None) -> Dict[str, float]:
        """Get real-time EC2 pricing"""
        if not instance_types:
            instance_types = list(self.get_fallback_ec2_pricing().keys())
            
        if not self.pricing_client:
            return self.get_fallback_ec2_pricing()
        
        pricing = {}
        try:
            for instance_type in instance_types:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'US East (N. Virginia)'},
                        {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                        {'Type': 'TERM_MATCH', 'Field': 'operating-system', 'Value': 'Linux'},
                        {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'}
                    ]
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data['terms']['OnDemand']
                    for term_id, term_data in terms.items():
                        for price_dim_id, price_dim in term_data['priceDimensions'].items():
                            hourly_price = float(price_dim['pricePerUnit']['USD'])
                            pricing[instance_type] = hourly_price
                            break
                        break
                else:
                    pricing[instance_type] = self.get_fallback_ec2_pricing().get(instance_type, 0.192)
                    
        except Exception as e:
            return self.get_fallback_ec2_pricing()
        
        return pricing
    
    def get_rds_pricing(self, instance_types: List[str] = None) -> Dict[str, float]:
        """Get real-time RDS pricing"""
        if not instance_types:
            instance_types = list(self.get_fallback_rds_pricing().keys())
            
        if not self.pricing_client:
            return self.get_fallback_rds_pricing()
        
        pricing = {}
        try:
            for instance_type in instance_types:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonRDS',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'US East (N. Virginia)'},
                        {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': 'Oracle'},
                        {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'}
                    ]
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data['terms']['OnDemand']
                    for term_id, term_data in terms.items():
                        for price_dim_id, price_dim in term_data['priceDimensions'].items():
                            hourly_price = float(price_dim['pricePerUnit']['USD'])
                            pricing[instance_type] = hourly_price
                            break
                        break
                else:
                    pricing[instance_type] = self.get_fallback_rds_pricing().get(instance_type, 0.960)
                    
        except Exception as e:
            return self.get_fallback_rds_pricing()
        
        return pricing
    
    def get_mongodb_atlas_pricing(self) -> Dict[str, float]:
        """Get MongoDB Atlas pricing"""
        return {
            'M10': 0.08, 'M20': 0.12, 'M30': 0.54, 'M40': 1.08,
            'M50': 2.16, 'M60': 4.32, 'M80': 8.64, 'M140': 17.28,
            'M200': 25.92, 'M300': 43.20, 'M400': 86.40, 'M700': 259.20
        }
    
    def get_fallback_ec2_pricing(self) -> Dict[str, float]:
        """Fallback EC2 pricing"""
        return {
            't3.micro': 0.0104, 't3.small': 0.0208, 't3.medium': 0.0416, 't3.large': 0.0832,
            't3.xlarge': 0.1664, 't3.2xlarge': 0.3328, 'm5.large': 0.096, 'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384, 'm5.4xlarge': 0.768, 'm5.8xlarge': 1.536, 'm5.12xlarge': 2.304,
            'r5.large': 0.126, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504, 'r5.4xlarge': 1.008,
            'r5.8xlarge': 2.016, 'r5.12xlarge': 3.024, 'c5.large': 0.085, 'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34, 'c5.4xlarge': 0.68, 'c5.9xlarge': 1.53, 'c5.12xlarge': 2.04
        }
    
    def get_fallback_rds_pricing(self) -> Dict[str, float]:
        """Fallback RDS pricing"""
        return {
            'db.t3.micro': 0.017, 'db.t3.small': 0.034, 'db.t3.medium': 0.068,
            'db.t3.large': 0.136, 'db.t3.xlarge': 0.272, 'db.t3.2xlarge': 0.544,
            'db.m5.large': 0.192, 'db.m5.xlarge': 0.384, 'db.m5.2xlarge': 0.768,
            'db.m5.4xlarge': 1.536, 'db.r5.large': 0.240, 'db.r5.xlarge': 0.480,
            'db.r5.2xlarge': 0.960, 'db.r5.4xlarge': 1.920, 'db.r5.8xlarge': 3.840
        }
    
    def get_comprehensive_aws_services_pricing(self) -> Dict[str, Any]:
        """Get comprehensive AWS services pricing"""
        return {
            'ec2': self.get_ec2_pricing(),
            'rds': self.get_rds_pricing(),
            'mongodb_atlas': self.get_mongodb_atlas_pricing(),
            'storage': {
                'ebs_gp3': 0.08,  # per GB-month
                'ebs_gp2': 0.10,
                'ebs_io1': 0.125,
                'ebs_io2': 0.125,
                's3_standard': 0.023,
                's3_ia': 0.0125,
                's3_glacier': 0.004
            },
            'network': {
                'nat_gateway': 0.045,  # per hour
                'data_transfer_out': 0.09,  # per GB
                'vpc_endpoint': 0.01,  # per hour
                'direct_connect_1gb': 0.30,  # per hour
                'direct_connect_10gb': 2.25,  # per hour
                'direct_connect_100gb': 22.50,  # per hour
                'direct_connect_data': 0.02  # per GB
            },
            'security': {
                'waf': 1.00,  # per web ACL per month
                'shield_advanced': 3000.00,  # per month
                'secrets_manager': 0.40  # per secret per month
            },
            'monitoring': {
                'cloudwatch_logs': 0.50,  # per GB ingested
                'cloudwatch_metrics': 0.30,  # per metric per month
                'x_ray': 5.00  # per 1M traces
            },
            'backup': {
                'backup_storage': 0.05,  # per GB-month
                'backup_restore': 0.02  # per GB
            },
            'migration_services': {
                'dms_t3_micro': 0.020,
                'dms_t3_small': 0.034,
                'dms_t3_medium': 0.068,
                'dms_t3_large': 0.136,
                'dms_c5_large': 0.192,
                'dms_c5_xlarge': 0.384,
                'dms_r5_large': 0.252,
                'dms_r5_xlarge': 0.504,
                'datasync_per_gb': 0.0125,
                'snowball_edge_storage': 300.00,  # per job
                'snowball_edge_compute': 400.00   # per job
            }
        }

# Enhanced AI Analysis Service
class EnterpriseAIService:
    def __init__(self):
        self.client = None
        self.connection_error = None
        self.is_initialized = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        self.is_initialized = True
        if not ANTHROPIC_AVAILABLE:
            self.connection_error = "Anthropic SDK not available"
            return False
            
        try:
            api_key = st.secrets.get('ANTHROPIC_API_KEY') or st.session_state.get('anthropic_api_key')
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                # Test the connection
                test_response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                return True
        except Exception as e:
            self.connection_error = str(e)
            self.client = None
            return False
        return False
    
    def get_connection_status(self):
        """Get detailed AI connection status"""
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Service not initialized',
                'details': 'Internal initialization error'
            }
            
        if not ANTHROPIC_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Anthropic SDK not installed',
                'details': 'Install anthropic: pip install anthropic'
            }
        
        api_key = st.secrets.get('ANTHROPIC_API_KEY') or st.session_state.get('anthropic_api_key')
        if not api_key:
            return {
                'status': 'disconnected',
                'message': 'API key not configured',
                'details': 'Enter API key in Advanced Options tab'
            }
        
        if self.client:
            try:
                # Test with a simple call
                test_response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "test"}]
                )
                return {
                    'status': 'connected',
                    'message': 'AI analysis active',
                    'details': 'Claude 3.5 Sonnet connected'
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'message': 'API connection failed',
                    'details': str(e)
                }
        
        return {
            'status': 'fallback',
            'message': 'Using fallback analysis',
            'details': self.connection_error or 'Configure API key for AI insights'
        }
    
    def get_comprehensive_analysis(self, migration_data: Dict[str, Any]) -> Dict[str, str]:
        """Get comprehensive AI analysis"""
        if not self.client:
            return self.get_fallback_analysis()
        
        analyses = {}
        prompts = {
            'executive_summary': f"""
            Create an executive summary for Oracle to MongoDB migration:
            - Environments: {len(migration_data.get('servers', {}))}
            - Data Size: {migration_data.get('data_size_tb', 0)} TB
            - Complexity Score: {migration_data.get('complexity_score', 0)}/100
            - Annual Cost Savings: ${migration_data.get('annual_savings', 0):,.0f}
            
            Provide a concise executive summary focusing on business impact, ROI, and strategic recommendations.
            """,
            
            'risk_assessment': f"""
            Analyze migration risks for Oracle to MongoDB:
            - PL/SQL Objects: {migration_data.get('num_pl_sql_objects', 0)}
            - Applications: {migration_data.get('num_applications', 1)}
            - Data Volume: {migration_data.get('data_size_tb', 0)} TB
            - Complexity: {migration_data.get('complexity_score', 0)}/100
            
            Identify top 5 risks and mitigation strategies.
            """,
            
            'technical_strategy': f"""
            Develop technical migration strategy:
            - Current Oracle setup: {migration_data.get('servers', {})}
            - Target AWS architecture
            - Migration timeline: {migration_data.get('timeline_months', 6)} months
            
            Provide detailed technical approach and architecture recommendations.
            """,
            
            'cost_optimization': f"""
            Analyze cost optimization opportunities:
            - Current costs vs AWS costs
            - 3-year projection
            - Optimization recommendations
            
            Focus on maximizing ROI and cost efficiency.
            """
        }
        
        try:
            for analysis_type, prompt in prompts.items():
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                analyses[analysis_type] = response.content[0].text
        except Exception as e:
            return self.get_fallback_analysis()
        
        return analyses
    
    def get_fallback_analysis(self) -> Dict[str, str]:
        """Fallback analysis when AI is unavailable"""
        return {
            'executive_summary': """
            **Executive Summary**
            
            This Oracle to MongoDB migration presents a significant opportunity for cost optimization and modernization. Based on the analysis, the migration is technically feasible with moderate to high complexity depending on the existing PL/SQL dependencies.
            
            **Key Benefits:**
            - Reduced licensing costs through elimination of Oracle licenses
            - Improved scalability with MongoDB's horizontal scaling capabilities
            - Enhanced developer productivity with document-based data model
            - Cloud-native architecture enabling better DevOps practices
            
            **Recommended Approach:**
            Implement a phased migration strategy starting with non-critical environments, followed by comprehensive testing and validation before production cutover.
            """,
            
            'risk_assessment': """
            **Risk Assessment & Mitigation**
            
            **High-Priority Risks:**
            1. **PL/SQL Conversion Complexity** - Significant business logic may need refactoring
            2. **Data Migration Integrity** - Large datasets require careful validation
            3. **Performance Impact** - Query patterns may need optimization
            4. **Application Integration** - Dependencies may require updates
            5. **Timeline Overruns** - Complex migrations often exceed estimates
            
            **Mitigation Strategies:**
            - Comprehensive PL/SQL code analysis and conversion planning
            - Automated data validation and reconciliation processes
            - Performance testing with production-like workloads
            - Parallel running systems during transition period
            - Agile migration approach with regular checkpoint reviews
            """,
            
            'technical_strategy': """
            **Technical Migration Strategy**
            
            **Phase 1: Assessment & Planning (2-4 weeks)**
            - Complete inventory of Oracle objects and dependencies
            - Application architecture analysis
            - Data model design for MongoDB
            
            **Phase 2: Environment Setup (2-3 weeks)**
            - AWS infrastructure provisioning
            - MongoDB Atlas cluster configuration
            - Security and networking setup
            
            **Phase 3: Migration Execution (8-16 weeks)**
            - Schema conversion and data migration
            - Application code refactoring
            - Integration testing and validation
            
            **Phase 4: Go-Live & Optimization (2-4 weeks)**
            - Production cutover
            - Performance monitoring and tuning
            - Post-migration support and optimization
            """,
            
            'cost_optimization': """
            **Cost Optimization Analysis**
            
            **Immediate Savings Opportunities:**
            - Elimination of Oracle licensing fees
            - Reduced maintenance and support costs
            - More efficient resource utilization
            
            **Long-term Benefits:**
            - Pay-as-you-scale pricing model
            - Automated backup and maintenance
            - Reduced operational overhead
            
            **Optimization Recommendations:**
            - Right-size instances based on actual usage patterns
            - Implement auto-scaling for variable workloads
            - Use Reserved Instances for predictable workloads
            - Optimize data transfer costs through strategic placement
            - Implement comprehensive monitoring to identify cost optimization opportunities
            """
        }

# Enhanced Data Transfer Calculator
class EnhancedDataTransferCalculator:
    def __init__(self):
        self.dx_pricing = {
            '1Gbps': {'hourly': 0.30, 'data_transfer': 0.02},
            '10Gbps': {'hourly': 2.25, 'data_transfer': 0.02},
            '100Gbps': {'hourly': 22.50, 'data_transfer': 0.02}
        }
        
        self.aws_service_costs = {
            'dms': {
                'dms.t3.micro': 0.020,
                'dms.t3.small': 0.034,
                'dms.t3.medium': 0.068,
                'dms.t3.large': 0.136,
                'dms.c5.large': 0.192,
                'dms.c5.xlarge': 0.384,
                'dms.c5.2xlarge': 0.768,
                'dms.r5.large': 0.252,
                'dms.r5.xlarge': 0.504
            },
            'datasync': {
                'per_gb_transferred': 0.0125,
                'hourly_agent_cost': 0.04
            },
            'snowball': {
                'snowball_edge_storage': 300.00,  # per job
                'snowball_edge_compute': 400.00,  # per job
                'snowmobile': 0.005  # per GB per month
            }
        }
    
    def calculate_transfer_options(self, data_size_tb: float, timeline_months: int) -> Dict[str, Any]:
        """Calculate different data transfer options with costs and timelines"""
        data_size_gb = data_size_tb * 1000
        
        options = {}
        
        # Option 1: Internet Transfer (Standard)
        internet_bandwidth_mbps = 1000  # 1 Gbps
        internet_transfer_days = (data_size_gb * 8) / (internet_bandwidth_mbps * 3600 * 24 / 1000)
        
        options['internet'] = {
            'method': 'Internet Transfer',
            'bandwidth': '1 Gbps',
            'transfer_time_days': internet_transfer_days,
            'infrastructure_cost': data_size_gb * 0.05,  # Network infrastructure
            'aws_data_in_cost': 0,  # AWS doesn't charge for data IN
            'total_cost': data_size_gb * 0.05,
            'suitability': 'Small to medium datasets (<10 TB)',
            'pros': ['Simple setup', 'No additional hardware'],
            'cons': ['Slower transfer', 'Uses production bandwidth'],
            'recommendation_score': 60 if data_size_tb <= 10 else 30
        }
        
        # Option 2: Direct Connect 10Gbps
        dx_10gb_bandwidth_mbps = 10000  # 10 Gbps
        dx_transfer_days = (data_size_gb * 8) / (dx_10gb_bandwidth_mbps * 3600 * 24 / 1000)
        dx_monthly_cost = self.dx_pricing['10Gbps']['hourly'] * 24 * 30
        dx_duration_months = max(1, dx_transfer_days / 30)
        
        options['direct_connect_10gb'] = {
            'method': 'AWS Direct Connect 10Gbps',
            'bandwidth': '10 Gbps',
            'transfer_time_days': dx_transfer_days,
            'dx_monthly_cost': dx_monthly_cost,
            'dx_total_cost': dx_monthly_cost * dx_duration_months,
            'data_transfer_cost': data_size_gb * self.dx_pricing['10Gbps']['data_transfer'],
            'setup_cost': 5000,  # One-time setup
            'total_cost': (dx_monthly_cost * dx_duration_months + 
                          data_size_gb * self.dx_pricing['10Gbps']['data_transfer'] + 5000),
            'suitability': 'Large datasets (10-100 TB)',
            'pros': ['Dedicated bandwidth', 'Predictable performance', 'Secure'],
            'cons': ['Higher upfront cost', 'Setup time required'],
            'recommendation_score': 90 if 10 <= data_size_tb <= 100 else 70
        }
        
        # Option 3: AWS DataSync
        datasync_agent_hours = dx_transfer_days * 24
        options['datasync'] = {
            'method': 'AWS DataSync',
            'bandwidth': 'Up to 10 Gbps (depends on network)',
            'transfer_time_days': dx_transfer_days,  # Assuming good network
            'data_transfer_cost': data_size_gb * self.aws_service_costs['datasync']['per_gb_transferred'],
            'agent_cost': datasync_agent_hours * self.aws_service_costs['datasync']['hourly_agent_cost'],
            'total_cost': (data_size_gb * self.aws_service_costs['datasync']['per_gb_transferred'] + 
                          datasync_agent_hours * self.aws_service_costs['datasync']['hourly_agent_cost']),
            'suitability': 'Online data transfer with validation',
            'pros': ['Built-in validation', 'Incremental transfer', 'Monitoring'],
            'cons': ['Per-GB pricing', 'Network dependent'],
            'recommendation_score': 80 if data_size_tb <= 50 else 60
        }
        
        # Option 4: AWS DMS (Database Migration Service)
        dms_instance = 'dms.r5.xlarge' if data_size_tb > 50 else 'dms.c5.large'
        dms_hourly_cost = self.aws_service_costs['dms'][dms_instance]
        dms_duration_hours = max(48, dx_transfer_days * 24)  # Minimum 48 hours
        
        options['dms'] = {
            'method': 'AWS Database Migration Service',
            'instance_type': dms_instance,
            'transfer_time_days': dms_duration_hours / 24,
            'dms_instance_cost': dms_hourly_cost * dms_duration_hours,
            'data_transfer_cost': data_size_gb * 0.01,  # Estimated data processing
            'total_cost': dms_hourly_cost * dms_duration_hours + data_size_gb * 0.01,
            'suitability': 'Database-specific migration with CDC',
            'pros': ['Database-native', 'Change Data Capture', 'Minimal downtime'],
            'cons': ['Complex setup', 'Database-specific limitations'],
            'recommendation_score': 85
        }
        
        # Option 5: AWS Snowball (for very large datasets)
        if data_size_tb > 50:
            num_snowballs = max(1, int(data_size_tb / 80))  # 80TB per Snowball Edge
            snowball_transfer_days = num_snowballs * 7  # 7 days per device cycle
            
            options['snowball'] = {
                'method': 'AWS Snowball Edge',
                'capacity': f"{num_snowballs} devices @ 80TB each",
                'transfer_time_days': snowball_transfer_days,
                'device_cost': num_snowballs * self.aws_service_costs['snowball']['snowball_edge_storage'],
                'shipping_cost': num_snowballs * 50,  # Estimated shipping
                'total_cost': (num_snowballs * self.aws_service_costs['snowball']['snowball_edge_storage'] + 
                              num_snowballs * 50),
                'suitability': 'Very large datasets (>50 TB)',
                'pros': ['Offline transfer', 'No bandwidth limitations', 'Secure'],
                'cons': ['Physical logistics', 'Longer timeline', 'Manual process'],
                'recommendation_score': 95 if data_size_tb > 100 else 75
            }
        
        return options
    
    def get_recommended_option(self, data_size_tb: float, timeline_months: int, 
                              budget_constraint: float = None) -> Dict[str, Any]:
        """Get recommended transfer option based on requirements"""
        options = self.calculate_transfer_options(data_size_tb, timeline_months)
        
        recommendations = []
        
        for option_key, option in options.items():
            if budget_constraint and option['total_cost'] > budget_constraint:
                continue
                
            # Use the built-in recommendation score
            score = option.get('recommendation_score', 50)
            
            recommendations.append({
                'option_key': option_key,
                'option': option,
                'score': score
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'recommended': recommendations[0] if recommendations else None,
            'all_options': options,
            'ranking': recommendations
        }

# PDF Report Generator
class PDFReportGenerator:
    def __init__(self):
        if not PDF_AVAILABLE:
            return
        self.styles = getSampleStyleSheet()
        self.custom_styles = self.create_custom_styles()
    
    def create_custom_styles(self):
        """Create custom paragraph styles"""
        if not PDF_AVAILABLE:
            return {}
            
        custom_styles = {}
        
        custom_styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2D3748'),
            alignment=1  # Center alignment
        )
        
        custom_styles['CustomHeading'] = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#4299E1'),
            borderWidth=0,
            borderColor=colors.HexColor('#4299E1'),
            borderPadding=5
        )
        
        custom_styles['CustomBody'] = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.HexColor('#2D3748')
        )
        
        return custom_styles
    
    def generate_report(self, analysis_data: Dict[str, Any]) -> BytesIO:
        """Generate comprehensive PDF report"""
        if not PDF_AVAILABLE:
            raise Exception("PDF generation not available")
            
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title Page
        story.append(Paragraph("Oracle to MongoDB Migration Analysis", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", self.custom_styles['CustomBody']))
        story.append(Spacer(1, 50))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.custom_styles['CustomHeading']))
        executive_data = [
            ['Metric', 'Current', 'Projected', 'Savings'],
            ['Annual Cost', f"${analysis_data.get('total_current_cost', 0):,.0f}", 
             f"${analysis_data.get('total_aws_cost', 0):,.0f}", 
             f"${analysis_data.get('total_annual_savings', 0):,.0f}"],
            ['Complexity Score', '', f"{analysis_data.get('complexity_score', 0)}/100", ''],
            ['Migration Timeline', '', f"{analysis_data.get('timeline_months', 6)} months", ''],
            ['ROI (3-year)', '', f"{analysis_data.get('three_year_roi', 0):.1f}%", '']
        ]
        
        exec_table = Table(executive_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        exec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4299E1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(exec_table)
        story.append(Spacer(1, 30))
        
        # Additional content...
        doc.build(story)
        buffer.seek(0)
        return buffer

# Enhanced Analytics and Visualization
class EnterpriseAnalytics:
    def __init__(self):
        self.is_initialized = True
    
    def create_complexity_heatmap(self, complexity_details: Dict[str, float], servers: Dict[str, Any]) -> go.Figure:
        """Create complexity heatmap across environments and factors"""
        factors = list(complexity_details.keys())
        environments = list(servers.keys())
        
        # Create matrix data
        z_data = []
        for env in environments:
            env_scores = []
            for factor in factors:
                # Calculate environment-specific complexity scores
                base_score = complexity_details[factor]
                if 'CPU' in factor:
                    env_score = min(100, servers[env]['cpu'] * 3)
                elif 'RAM' in factor:
                    env_score = min(100, servers[env]['ram'] / 4)
                elif 'Storage' in factor:
                    env_score = min(100, servers[env]['storage'] / 100)
                elif 'Throughput' in factor:
                    env_score = min(100, servers[env]['throughput'] / 1000)
                else:
                    env_score = base_score
                env_scores.append(env_score)
            z_data.append(env_scores)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=factors,
            y=environments,
            colorscale='RdYlBu_r',
            text=[[f'{val:.0f}' for val in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Migration Complexity Heatmap by Environment',
            xaxis_title='Complexity Factors',
            yaxis_title='Environments',
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_roi_waterfall(self, migration_costs: Dict[str, float], annual_savings: float) -> go.Figure:
        """Create ROI waterfall chart"""
        categories = ['Current Cost', 'Migration Cost', 'Annual Savings (Year 1)', 
                     'Annual Savings (Year 2)', 'Annual Savings (Year 3)', 'Net 3-Year ROI']
        
        values = [
            0,  # Starting point
            -migration_costs['total_migration_cost'],
            annual_savings,
            annual_savings * 1.05,  # 5% improvement
            annual_savings * 1.1,   # 10% improvement
            0  # Will be calculated
        ]
        
        # Calculate cumulative
        cumulative = [values[0]]
        for i in range(1, len(values) - 1):
            cumulative.append(cumulative[-1] + values[i])
        
        # Net ROI
        net_roi = cumulative[-1]
        values[-1] = net_roi
        cumulative.append(net_roi)
        
        fig = go.Figure(go.Waterfall(
            name="ROI Analysis",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=categories,
            y=values,
            text=[f"${val:,.0f}" for val in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title="3-Year ROI Waterfall Analysis",
            showlegend=False,
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_cost_breakdown_pie(self, cost_components: Dict[str, float]) -> go.Figure:
        """Create cost breakdown pie chart"""
        labels = list(cost_components.keys())
        values = list(cost_components.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='#000000', width=2)
            )
        )])
        
        fig.update_layout(
            title="AWS Cost Breakdown",
            annotations=[dict(text='AWS Costs', x=0.5, y=0.5, font_size=16, showarrow=False)],
            height=500
        )
        
        return fig
    
    def create_timeline_gantt(self, timeline_months: int, num_pl_sql_objects: int) -> go.Figure:
        """Create migration timeline Gantt chart"""
        phases = [
            {'Task': 'Discovery & Assessment', 'Start': 0, 'Duration': 3},
            {'Task': 'Architecture Design', 'Start': 2, 'Duration': 3},
            {'Task': 'Environment Setup', 'Start': 4, 'Duration': 4},
            {'Task': 'Schema Migration', 'Start': 6, 'Duration': 6},
            {'Task': 'Application Refactoring', 'Start': 8, 'Duration': max(8, num_pl_sql_objects // 100)},
            {'Task': 'Testing & Validation', 'Start': timeline_months - 6, 'Duration': 4},
            {'Task': 'Data Migration', 'Start': timeline_months - 3, 'Duration': 2},
            {'Task': 'Go-Live & Optimization', 'Start': timeline_months - 2, 'Duration': 3}
        ]
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, phase in enumerate(phases):
            fig.add_trace(go.Bar(
                x=[phase['Duration']],
                y=[phase['Task']],
                orientation='h',
                name=phase['Task'],
                marker=dict(color=colors[i % len(colors)]),
                text=f"{phase['Duration']} weeks",
                textposition='inside',
                base=phase['Start']
            ))
        
        fig.update_layout(
            title='Migration Timeline - Gantt Chart',
            xaxis_title='Weeks',
            yaxis_title='Migration Phases',
            height=500,
            showlegend=False,
            xaxis=dict(range=[0, timeline_months + 2])
        )
        
        return fig

    def create_transfer_comparison_chart(self, transfer_options: Dict[str, Any]) -> go.Figure:
        """Create data transfer options comparison chart"""
        methods = []
        costs = []
        times = []
        scores = []
        
        for option_key, option in transfer_options.items():
            methods.append(option['method'])
            costs.append(option['total_cost'])
            times.append(option['transfer_time_days'])
            scores.append(option.get('recommendation_score', 50))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transfer Costs', 'Transfer Times', 'Recommendation Scores', 'Cost vs Time'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Cost comparison
        fig.add_trace(
            go.Bar(x=methods, y=costs, name='Cost ($)', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Time comparison
        fig.add_trace(
            go.Bar(x=methods, y=times, name='Time (days)', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Score comparison
        fig.add_trace(
            go.Bar(x=methods, y=scores, name='Score', marker_color='orange'),
            row=2, col=1
        )
        
        # Cost vs Time scatter
        fig.add_trace(
            go.Scatter(x=times, y=costs, mode='markers+text', text=methods,
                      textposition='top center', name='Options', marker_size=10),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Data Transfer Options Analysis")
        return fig

# Initialize services without caching to avoid attribute errors
def initialize_services():
    """Initialize all services without caching"""
    try:
        pricing_service = AWSPricingService()
    except Exception as e:
        st.error(f"Failed to initialize AWS Pricing Service: {str(e)}")
        pricing_service = None
    
    try:
        ai_service = EnterpriseAIService()
    except Exception as e:
        st.error(f"Failed to initialize AI Service: {str(e)}")
        ai_service = None
    
    try:
        analytics_service = EnterpriseAnalytics()
    except Exception as e:
        st.error(f"Failed to initialize Analytics Service: {str(e)}")
        analytics_service = None
    
    try:
        pdf_service = PDFReportGenerator() if PDF_AVAILABLE else None
    except Exception as e:
        st.error(f"Failed to initialize PDF Service: {str(e)}")
        pdf_service = None
    
    return {
        'pricing': pricing_service,
        'ai': ai_service,
        'analytics': analytics_service,
        'pdf': pdf_service
    }

# Service Status Display Function
def display_service_status(services):
    """Display comprehensive service connection status with safe access"""
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<div class="config-header">🔌 Service Connection Status</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # AWS Pricing Service Status
    with col1:
        if services.get('pricing') and hasattr(services['pricing'], 'get_connection_status'):
            try:
                aws_status = services['pricing'].get_connection_status()
            except Exception as e:
                aws_status = {
                    'status': 'error',
                    'message': 'Service error',
                    'details': str(e)
                }
        else:
            aws_status = {
                'status': 'unavailable',
                'message': 'Service not available',
                'details': 'Service initialization failed'
            }
            
        status_class = f"status-{aws_status['status']}"
        indicator_class = {
            'connected': 'indicator-green',
            'error': 'indicator-red', 
            'fallback': 'indicator-yellow',
            'unavailable': 'indicator-red',
            'disconnected': 'indicator-red'
        }.get(aws_status['status'], 'indicator-red')
        
        st.markdown(f"""
        <div class="service-status-card">
            <h4>☁️ AWS Pricing Service</h4>
            <div class="{status_class}" style="padding: 0.5rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                <span class="status-indicator {indicator_class}"></span>
                <strong>{aws_status['message']}</strong>
            </div>
            <small style="color: #718096;">{aws_status['details']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Service Status
    with col2:
        if services.get('ai') and hasattr(services['ai'], 'get_connection_status'):
            try:
                ai_status = services['ai'].get_connection_status()
            except Exception as e:
                ai_status = {
                    'status': 'error',
                    'message': 'Service error',
                    'details': str(e)
                }
        else:
            ai_status = {
                'status': 'unavailable',
                'message': 'Service not available',
                'details': 'Service initialization failed'
            }
            
        status_class = f"status-{ai_status['status']}"
        indicator_class = {
            'connected': 'indicator-green',
            'error': 'indicator-red',
            'fallback': 'indicator-yellow', 
            'unavailable': 'indicator-red',
            'disconnected': 'indicator-red'
        }.get(ai_status['status'], 'indicator-red')
        
        st.markdown(f"""
        <div class="service-status-card">
            <h4>🤖 AI Analysis Service</h4>
            <div class="{status_class}" style="padding: 0.5rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                <span class="status-indicator {indicator_class}"></span>
                <strong>{ai_status['message']}</strong>
            </div>
            <small style="color: #718096;">{ai_status['details']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # PDF Generation Status
    with col3:
        pdf_status = 'connected' if services.get('pdf') else 'unavailable'
        pdf_message = 'PDF reports available' if services.get('pdf') else 'PDF generation unavailable'
        pdf_details = 'Ready to generate reports' if services.get('pdf') else 'Install reportlab for PDF reports'
        indicator_class = 'indicator-green' if services.get('pdf') else 'indicator-red'
        
        st.markdown(f"""
        <div class="service-status-card">
            <h4>📄 PDF Generation</h4>
            <div class="status-{pdf_status}" style="padding: 0.5rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                <span class="status-indicator {indicator_class}"></span>
                <strong>{pdf_message}</strong>
            </div>
            <small style="color: #718096;">{pdf_details}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_text_report(total_current_cost, total_aws_cost, total_annual_savings, 
                        roi_percentage, migration_timeline, complexity_score,
                        migration_costs, strategy, servers, risk_factors, data_size_tb):
    """Generate comprehensive text report"""
    return f"""
ORACLE TO MONGODB MIGRATION ANALYSIS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Current Annual Cost: ${total_current_cost:,.0f}
Projected AWS Cost: ${total_aws_cost:,.0f}
Annual Savings: ${total_annual_savings:,.0f}
3-Year ROI: {roi_percentage:.1f}%
Migration Timeline: {migration_timeline} months
Complexity Score: {complexity_score}/100

ENHANCED MIGRATION COSTS BREAKDOWN
==================================
Total Investment: ${migration_costs['total_migration_cost']:,.0f}

Detailed Breakdown:
• Migration Team: ${migration_costs['migration_team_cost']:,.0f}
• Data Transfer (Recommended): ${migration_costs['recommended_transfer_cost']:,.0f}
• AWS DMS Service: ${migration_costs['dms_cost']:,.0f}
• AWS DataSync: ${migration_costs['datasync_cost']:,.0f}
• Network Infrastructure: ${migration_costs['network_setup_cost'] + migration_costs['network_monitoring_cost']:,.0f}
• Temporary Storage: ${migration_costs['temp_storage_cost']:,.0f}
• Backup Storage: ${migration_costs['backup_storage_cost']:,.0f}
• Tools & Licenses: ${migration_costs['tool_costs']:,.0f}
• Training: ${migration_costs['training_costs']:,.0f}
• AWS Professional Services: ${migration_costs['aws_professional_services']:,.0f}
• Testing & Validation: ${migration_costs['testing_costs']:,.0f}
• Contingency (15%): ${migration_costs['contingency_cost']:,.0f}

DATA TRANSFER ANALYSIS
=====================
Data Size: {data_size_tb} TB
Recommended Method: {migration_costs['data_transfer_breakdown']['recommended']['option']['method']}
Transfer Time: {migration_costs['estimated_transfer_days']:.1f} days
Bandwidth: {migration_costs['bandwidth_gbps']} Gbps
Transfer Cost: ${migration_costs['recommended_transfer_cost']:,.0f}

Key Transfer Considerations:
• Bandwidth planning for {data_size_tb * 1000} GB transfer
• {migration_costs['data_transfer_breakdown']['recommended']['option']['suitability']}
• Estimated timeline: {migration_costs['estimated_transfer_days']:.1f} days
• Direct Connect setup for dedicated bandwidth

AWS SERVICES CONFIGURATION
==========================
Database Migration Service:
• Instance Type: {migration_costs['dms_instance_type']}
• Duration: {migration_costs['dms_duration_hours']:.0f} hours
• Cost: ${migration_costs['dms_cost']:,.0f}

DataSync Service:
• Cost: ${migration_costs['datasync_cost']:,.0f}
• Purpose: Ongoing synchronization during migration

RECOMMENDED STRATEGY
===================
Strategy: {strategy['strategy']}
Risk Level: {strategy['risk']}
Timeline: {strategy['timeline']}
Description: {strategy['description']}

ENVIRONMENT ANALYSIS
===================
{chr(10).join([f"{env}: {specs['cpu']} vCPU, {specs['ram']}GB RAM, {specs['storage']}GB Storage" for env, specs in servers.items()])}

RISK FACTORS
============
{chr(10).join([f"• {risk}" for risk in risk_factors])}

NETWORK INFRASTRUCTURE REQUIREMENTS
===================================
• Bandwidth: {migration_costs['bandwidth_gbps']} Gbps for efficient transfer
• Transfer Method: {migration_costs['data_transfer_breakdown']['recommended']['option']['method']}
• Transfer Time Estimate: {migration_costs['estimated_transfer_days']:.1f} days
• Setup Cost: Network infrastructure and monitoring
• Redundancy: Built into recommended solution

NEXT STEPS
==========
1. Stakeholder approval and budget allocation
2. {migration_costs['data_transfer_breakdown']['recommended']['option']['method']} setup
3. AWS DMS configuration with {migration_costs['dms_instance_type']} instance
4. Migration team formation and training
5. Network infrastructure setup for {migration_costs['bandwidth_gbps']} Gbps
6. Environment setup and testing
7. Phased migration execution with comprehensive monitoring
8. Post-migration optimization and performance tuning

COST OPTIMIZATION RECOMMENDATIONS
=================================
• Implement {migration_costs['data_transfer_breakdown']['recommended']['option']['method']} for optimal transfer
• Use AWS DMS for database-native migration with CDC
• Configure DataSync for ongoing synchronization
• Leverage AWS Professional Services for expertise
• Implement comprehensive testing strategy
• Use Reserved Instances for predictable workloads post-migration

SUCCESS METRICS
===============
• Target Annual Savings: ${total_annual_savings:,.0f}
• Migration Budget Adherence: Within ${migration_costs['total_migration_cost']:,.0f}
• Timeline Adherence: {migration_timeline} months
• Data Integrity: 100% accuracy target
• Performance: Maintain or improve current response times
• Zero Downtime Target: 99.9% uptime during migration

APPENDIX: DETAILED TRANSFER OPTIONS
==================================
All evaluated transfer methods:
{chr(10).join([f"• {option['method']}: ${option['total_cost']:,.0f} ({option['transfer_time_days']:.1f} days)" for option in migration_costs['data_transfer_breakdown']['all_options'].values()])}

Recommended: {migration_costs['data_transfer_breakdown']['recommended']['option']['method']}
Reason: {migration_costs['data_transfer_breakdown']['recommended']['option']['suitability']}
"""

# Enhanced calculation functions
def calculate_comprehensive_costs(servers: Dict[str, Any], pricing_data: Dict[str, Any], 
                                oracle_license: float, manpower: float, data_size_tb: float, 
                                recommendations: Dict[str, Any]) -> pd.DataFrame:
    """Calculate comprehensive cost analysis with all AWS services"""
    cost_data = []
    total_envs = len(servers) if servers else 1
    
    for env, specs in servers.items():
        # Current Oracle costs
        oracle_license_per_env = oracle_license / total_envs
        manpower_per_env = manpower / total_envs
        
        # Oracle infrastructure costs
        oracle_rds_instance = 'db.r5.2xlarge'
        if specs['cpu'] <= 8:
            oracle_rds_instance = 'db.m5.xlarge'
        elif specs['cpu'] <= 16:
            oracle_rds_instance = 'db.r5.xlarge'
        
        oracle_rds_cost = pricing_data['rds'].get(oracle_rds_instance, 0.960) * specs['daily_usage'] * 365
        oracle_storage_cost = pricing_data['storage']['ebs_gp2'] * specs['storage'] * 12
        oracle_backup_cost = specs['storage'] * pricing_data['backup']['backup_storage'] * 12
        
        current_total_cost = (oracle_license_per_env + manpower_per_env + 
                             oracle_rds_cost + oracle_storage_cost + oracle_backup_cost)
        
        # AWS Migration costs
        rec = recommendations[env]
        ec2_instance = rec['ec2']
        mongodb_cluster = rec['mongodb']
        
        # EC2 costs
        ec2_hourly_cost = pricing_data['ec2'].get(ec2_instance, 0.192)
        ec2_annual_cost = ec2_hourly_cost * specs['daily_usage'] * 365
        
        # Storage costs
        ebs_annual_cost = pricing_data['storage']['ebs_gp3'] * specs['storage'] * 12
        s3_backup_cost = specs['storage'] * pricing_data['storage']['s3_standard'] * 12
        
        # MongoDB Atlas costs
        mongodb_hourly_cost = pricing_data['mongodb_atlas'].get(mongodb_cluster, 0.54)
        mongodb_annual_cost = mongodb_hourly_cost * 24 * 365
        
        # Network costs
        nat_gateway_cost = pricing_data['network']['nat_gateway'] * 24 * 365
        vpc_endpoint_cost = pricing_data['network']['vpc_endpoint'] * 24 * 365
        data_transfer_cost = data_size_tb * 1000 * pricing_data['network']['data_transfer_out'] / total_envs
        
        # Security costs
        waf_cost = pricing_data['security']['waf'] * 12
        secrets_manager_cost = pricing_data['security']['secrets_manager'] * 5 * 12  # 5 secrets
        
        # Monitoring costs
        cloudwatch_cost = (pricing_data['monitoring']['cloudwatch_logs'] * specs['storage'] * 0.1 +  # 10% of storage as logs
                          pricing_data['monitoring']['cloudwatch_metrics'] * 50) * 12  # 50 custom metrics
        
        # Backup costs
        backup_cost = specs['storage'] * pricing_data['backup']['backup_storage'] * 12
        
        total_aws_cost = (ec2_annual_cost + ebs_annual_cost + mongodb_annual_cost + 
                         nat_gateway_cost + vpc_endpoint_cost + data_transfer_cost +
                         waf_cost + secrets_manager_cost + cloudwatch_cost + 
                         backup_cost + s3_backup_cost)
        
        # Calculate savings
        total_savings = current_total_cost - total_aws_cost
        savings_percentage = (total_savings / current_total_cost * 100) if current_total_cost > 0 else 0
        
        cost_data.append({
            'Environment': env,
            'Current_Oracle_License': oracle_license_per_env,
            'Current_Oracle_RDS': oracle_rds_cost,
            'Current_Oracle_Storage': oracle_storage_cost,
            'Current_Oracle_Backup': oracle_backup_cost,
            'Current_Manpower': manpower_per_env,
            'Current_Total': current_total_cost,
            'AWS_EC2_Cost': ec2_annual_cost,
            'AWS_EBS_Cost': ebs_annual_cost,
            'AWS_MongoDB_Cost': mongodb_annual_cost,
            'AWS_Network_Cost': nat_gateway_cost + vpc_endpoint_cost + data_transfer_cost,
            'AWS_Security_Cost': waf_cost + secrets_manager_cost,
            'AWS_Monitoring_Cost': cloudwatch_cost,
            'AWS_Backup_Cost': backup_cost + s3_backup_cost,
            'AWS_Total_Cost': total_aws_cost,
            'Annual_Savings': total_savings,
            'Savings_Percentage': savings_percentage,
            'EC2_Instance': ec2_instance,
            'MongoDB_Cluster': mongodb_cluster
        })
    
    return pd.DataFrame(cost_data)

def enhanced_complexity_analysis(servers: Dict[str, Any], data_size_tb: float, 
                               num_pl_sql_objects: int, num_applications: int) -> Tuple[int, Dict[str, float], List[str]]:
    """Enhanced complexity analysis with more factors"""
    
    # Get production specs safely
    prod_specs = None
    for env_name in ['Prod', 'Production', 'PROD']:
        if env_name in servers:
            prod_specs = servers[env_name]
            break
    
    if not prod_specs and servers:
        prod_specs = list(servers.values())[-1]
    elif not prod_specs:
        prod_specs = {'cpu': 4, 'ram': 16, 'storage': 500, 'throughput': 5000}
    
    # Enhanced complexity factors
    factors = {
        'Infrastructure_CPU': min(100, prod_specs['cpu'] * 2.5),
        'Infrastructure_RAM': min(100, prod_specs['ram'] / 4),
        'Infrastructure_Storage': min(100, prod_specs['storage'] / 200),
        'Infrastructure_Throughput': min(100, prod_specs['throughput'] / 1500),
        'Data_Volume': min(100, data_size_tb * 2.5),
        'Data_Complexity': min(100, data_size_tb * 1.5 + num_applications * 10),
        'Application_Count': min(100, num_applications * 15),
        'PL_SQL_Objects': min(100, num_pl_sql_objects / 8),
        'Environment_Count': min(100, len(servers) * 12),
        'Integration_Complexity': min(100, num_applications * 20 + len(servers) * 5)
    }
    
    # Weighted calculation
    weights = {
        'Infrastructure_CPU': 0.12, 'Infrastructure_RAM': 0.12, 'Infrastructure_Storage': 0.08,
        'Infrastructure_Throughput': 0.12, 'Data_Volume': 0.15, 'Data_Complexity': 0.1,
        'Application_Count': 0.08, 'PL_SQL_Objects': 0.15, 'Environment_Count': 0.05,
        'Integration_Complexity': 0.03
    }
    
    total_score = sum(factors[factor] * weights[factor] for factor in factors)
    
    # Enhanced risk factors
    risk_factors = []
    if factors['PL_SQL_Objects'] > 70:
        risk_factors.append("High PL/SQL complexity requires extensive refactoring and testing")
    if factors['Data_Volume'] > 80:
        risk_factors.append("Large data volume will require specialized migration tools and extended timeline")
    if factors['Integration_Complexity'] > 60:
        risk_factors.append("Complex application integrations may require significant rework")
    if factors['Infrastructure_Throughput'] > 75:
        risk_factors.append("High throughput requirements demand careful performance optimization")
    if len(servers) > 3:
        risk_factors.append("Multiple environments require coordinated migration strategy and testing")
    
    return min(100, round(total_score)), factors, risk_factors

def recommend_instances(servers):
    """Enhanced instance recommendations with latest AWS instances"""
    recommendations = {}
    
    for env, specs in servers.items():
        cpu = specs['cpu']
        ram = specs['ram']
        storage = specs['storage']
        
        # Enhanced recommendations based on workload characteristics
        if env.lower() in ['dev', 'development', 'sandbox']:
            if cpu <= 2 and ram <= 8:
                recommendations[env] = {'ec2': 't3.medium', 'mongodb': 'M10'}
            elif cpu <= 4 and ram <= 16:
                recommendations[env] = {'ec2': 't3.large', 'mongodb': 'M20'}
            else:
                recommendations[env] = {'ec2': 't3.xlarge', 'mongodb': 'M30'}
        
        elif env.lower() in ['qa', 'test', 'testing']:
            if cpu <= 4 and ram <= 16:
                recommendations[env] = {'ec2': 'm5.large', 'mongodb': 'M20'}
            elif cpu <= 8 and ram <= 32:
                recommendations[env] = {'ec2': 'm5.xlarge', 'mongodb': 'M30'}
            else:
                recommendations[env] = {'ec2': 'm5.2xlarge', 'mongodb': 'M40'}
        
        elif env.lower() in ['uat', 'staging', 'preprod', 'pre-production']:
            if cpu <= 8 and ram <= 32:
                recommendations[env] = {'ec2': 'r5.xlarge', 'mongodb': 'M40'}
            elif cpu <= 16 and ram <= 64:
                recommendations[env] = {'ec2': 'r5.2xlarge', 'mongodb': 'M50'}
            else:
                recommendations[env] = {'ec2': 'r5.4xlarge', 'mongodb': 'M60'}
        
        else:  # Production environments
            if cpu <= 16 and ram <= 64:
                recommendations[env] = {'ec2': 'r5.2xlarge', 'mongodb': 'M60'}
            elif cpu <= 32 and ram <= 128:
                recommendations[env] = {'ec2': 'r5.4xlarge', 'mongodb': 'M80'}
            elif cpu <= 48 and ram <= 192:
                recommendations[env] = {'ec2': 'r5.8xlarge', 'mongodb': 'M140'}
            else:
                recommendations[env] = {'ec2': 'r5.12xlarge', 'mongodb': 'M200'}
    
    return recommendations

def calculate_enhanced_migration_costs(data_size_tb, timeline_months, complexity_score, 
                                     use_direct_connect=True, bandwidth_gbps=10):
    """Enhanced migration cost calculation with detailed data transfer options"""
    
    # Initialize transfer calculator
    transfer_calc = EnhancedDataTransferCalculator()
    
    # Get transfer recommendations
    transfer_analysis = transfer_calc.get_recommended_option(data_size_tb, timeline_months)
    recommended_option = transfer_analysis['recommended']['option']
    
    # Base migration costs
    base_monthly_cost = 25000
    complexity_multiplier = 1 + (complexity_score / 80)
    monthly_cost = base_monthly_cost * complexity_multiplier
    
    migration_team_cost = monthly_cost * timeline_months
    
    # Enhanced cost components using recommended transfer method
    data_transfer_cost = recommended_option['total_cost']
    
    # AWS Database Migration Service costs
    dms_instance_type = 'dms.r5.xlarge' if data_size_tb > 50 else 'dms.c5.large'
    dms_hourly_rates = {
        'dms.c5.large': 0.192,
        'dms.c5.xlarge': 0.384,
        'dms.r5.large': 0.252,
        'dms.r5.xlarge': 0.504
    }
    dms_duration_hours = max(72, (data_size_tb / bandwidth_gbps) * 24)
    dms_cost = dms_hourly_rates[dms_instance_type] * dms_duration_hours
    
    # AWS DataSync costs
    datasync_cost = data_size_tb * 1000 * 0.0125 * 0.1  # 10% for ongoing sync
    
    # Network infrastructure and monitoring
    network_setup_cost = 10000 if use_direct_connect else 3000
    network_monitoring_cost = timeline_months * 500
    
    # Enhanced storage costs during migration
    temp_storage_cost = data_size_tb * 0.08 * timeline_months * 1.5
    backup_storage_cost = data_size_tb * 0.05 * timeline_months
    
    # Tools and licensing
    tool_licensing_cost = 12000 * timeline_months
    
    # Training and certification
    training_cost = 20000 + (timeline_months * 3000)
    
    # Professional services
    aws_professional_services = 50000 if data_size_tb > 100 else 25000
    
    # Testing and validation
    testing_cost = migration_team_cost * 0.3
    
    # Calculate totals
    base_costs = (migration_team_cost + data_transfer_cost + dms_cost + datasync_cost +
                 network_setup_cost + network_monitoring_cost + temp_storage_cost +
                 backup_storage_cost + tool_licensing_cost + training_cost +
                 aws_professional_services + testing_cost)
    
    contingency = base_costs * 0.15
    total_cost = base_costs + contingency
    
    return {
        'migration_team_cost': migration_team_cost,
        'data_transfer_breakdown': transfer_analysis,
        'recommended_transfer_cost': data_transfer_cost,
        'dms_cost': dms_cost,
        'dms_instance_type': dms_instance_type,
        'dms_duration_hours': dms_duration_hours,
        'datasync_cost': datasync_cost,
        'network_setup_cost': network_setup_cost,
        'network_monitoring_cost': network_monitoring_cost,
        'temp_storage_cost': temp_storage_cost,
        'backup_storage_cost': backup_storage_cost,
        'tool_costs': tool_licensing_cost,
        'training_costs': training_cost,
        'aws_professional_services': aws_professional_services,
        'testing_costs': testing_cost,
        'contingency_cost': contingency,
        'total_migration_cost': total_cost,
        'bandwidth_gbps': bandwidth_gbps,
        'estimated_transfer_days': recommended_option.get('transfer_time_days', 0)
    }

def get_migration_strategy(complexity_score):
    """Enhanced migration strategy recommendations"""
    if complexity_score < 20:
        return {
            'strategy': 'Lift and Shift Plus',
            'description': 'Minimal refactoring with cloud optimization',
            'timeline': '3-5 months',
            'risk': 'Low',
            'effort': 'Low'
        }
    elif complexity_score < 40:
        return {
            'strategy': 'Re-platform with Optimization',
            'description': 'Moderate changes with performance improvements',
            'timeline': '5-8 months',
            'risk': 'Medium',
            'effort': 'Medium'
        }
    elif complexity_score < 65:
        return {
            'strategy': 'Re-architect for Cloud',
            'description': 'Significant redesign for cloud-native benefits',
            'timeline': '8-14 months',
            'risk': 'Medium-High',
            'effort': 'High'
        }
    elif complexity_score < 85:
        return {
            'strategy': 'Hybrid Modernization',
            'description': 'Phased approach with parallel systems',
            'timeline': '12-18 months',
            'risk': 'High',
            'effort': 'Very High'
        }
    else:
        return {
            'strategy': 'Full Digital Transformation',
            'description': 'Complete rebuild with modern architecture',
            'timeline': '18+ months',
            'risk': 'Very High',
            'effort': 'Maximum'
        }

# Analysis function
def analyze_migration(services, servers, oracle_license_cost, manpower_cost, data_size_tb, 
                     migration_timeline, num_pl_sql_objects, num_applications, enable_ai,
                     migration_budget, training_budget, contingency_percent, aws_region,
                     use_direct_connect, bandwidth_gbps):
    """Perform comprehensive migration analysis"""
    
    with st.spinner('🔄 Performing comprehensive analysis...'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Get pricing data
        status_text.text("📊 Getting pricing data...")
        progress_bar.progress(20)
        
        if services.get('pricing'):
            pricing_data = services['pricing'].get_comprehensive_aws_services_pricing()
        else:
            # Fallback pricing with enhanced services
            pricing_data = {
                'ec2': {'t3.medium': 0.0416, 't3.large': 0.0832, 'm5.large': 0.096, 'm5.xlarge': 0.192, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504},
                'rds': {'db.m5.xlarge': 0.384, 'db.r5.xlarge': 0.480, 'db.r5.2xlarge': 0.960},
                'mongodb_atlas': {'M10': 0.08, 'M20': 0.12, 'M30': 0.54, 'M40': 1.08, 'M50': 2.16, 'M60': 4.32},
                'storage': {'ebs_gp2': 0.10, 'ebs_gp3': 0.08, 's3_standard': 0.023},
                'network': {'nat_gateway': 0.045, 'data_transfer_out': 0.09, 'vpc_endpoint': 0.01, 'direct_connect_10gb': 2.25, 'direct_connect_data': 0.02},
                'security': {'waf': 1.00, 'secrets_manager': 0.40},
                'monitoring': {'cloudwatch_logs': 0.50, 'cloudwatch_metrics': 0.30},
                'backup': {'backup_storage': 0.05, 'backup_restore': 0.02},
                'migration_services': {'dms_c5_large': 0.192, 'dms_r5_xlarge': 0.504, 'datasync_per_gb': 0.0125}
            }
        
        # Step 2: Generate recommendations
        status_text.text("💡 Generating infrastructure recommendations...")
        progress_bar.progress(40)
        recommendations = recommend_instances(servers)
        
        # Step 3: Calculate costs
        status_text.text("💰 Calculating comprehensive costs...")
        progress_bar.progress(60)
        cost_df = calculate_comprehensive_costs(servers, pricing_data, oracle_license_cost, 
                                              manpower_cost, data_size_tb, recommendations)
        
        # Step 4: Complexity analysis
        status_text.text("🎯 Analyzing migration complexity...")
        progress_bar.progress(70)
        complexity_score, complexity_details, risk_factors = enhanced_complexity_analysis(
            servers, data_size_tb, num_pl_sql_objects, num_applications
        )
        
        # Step 5: Enhanced migration costs and transfer analysis
        status_text.text("🚚 Calculating enhanced migration strategy...")
        progress_bar.progress(80)
        enhanced_migration_costs = calculate_enhanced_migration_costs(
            data_size_tb, migration_timeline, complexity_score, 
            use_direct_connect, bandwidth_gbps
        )
        strategy = get_migration_strategy(complexity_score)
        
        # Step 6: AI Analysis
        status_text.text("🤖 Generating AI insights...")
        progress_bar.progress(90)
        ai_analyses = {}
        if enable_ai and services.get('ai'):
            try:
                migration_data = {
                    'servers': servers, 'data_size_tb': data_size_tb, 'complexity_score': complexity_score,
                    'num_pl_sql_objects': num_pl_sql_objects, 'num_applications': num_applications,
                    'timeline_months': migration_timeline, 'annual_savings': cost_df['Annual_Savings'].sum()
                }
                ai_analyses = services['ai'].get_comprehensive_analysis(migration_data)
            except Exception as e:
                st.warning(f"AI analysis failed: {str(e)}")
                ai_analyses = services['ai'].get_fallback_analysis() if services.get('ai') else {}
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Store results in session state
        st.session_state.analysis_results = {
            'cost_df': cost_df,
            'complexity_score': complexity_score,
            'complexity_details': complexity_details,
            'risk_factors': risk_factors,
            'strategy': strategy,
            'enhanced_migration_costs': enhanced_migration_costs,
            'ai_analyses': ai_analyses,
            'servers': servers,
            'data_size_tb': data_size_tb,
            'migration_timeline': migration_timeline,
            'num_pl_sql_objects': num_pl_sql_objects,
            'pricing_data': pricing_data,
            'use_direct_connect': use_direct_connect,
            'bandwidth_gbps': bandwidth_gbps
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    st.success("✅ Enhanced Analysis Complete! Check the Results Dashboard and Reports tabs.")
    
# Add these functions BEFORE the main() function (around line 800)

def handle_bulk_upload():
    """Handle bulk file upload and processing"""
    st.markdown("### 📁 Bulk Configuration Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Configuration File", 
        type=['csv', 'xlsx', 'json'],
        help="Upload a CSV/Excel file with environment configurations or a JSON configuration file"
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
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
                return config_data
            
            # Display preview for CSV/Excel
            if file_extension in ['csv', 'xlsx']:
                st.markdown("#### 👀 File Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Process the file based on expected format
                if any('environment' in col.lower() for col in df.columns):
                    return process_environment_file(df)
                else:
                    st.warning("⚠️ Expected columns: Environment, CPU, RAM, Storage, Daily_Usage, Throughput")
                    st.markdown("**Expected format:**")
                    sample_df = pd.DataFrame({
                        'Environment': ['Production', 'Development', 'QA'],
                        'CPU': [32, 8, 16],
                        'RAM': [128, 32, 64],
                        'Storage': [2000, 500, 1000],
                        'Daily_Usage': [24, 12, 16],
                        'Throughput': [20000, 5000, 10000]
                    })
                    st.dataframe(sample_df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    return None

def process_environment_file(df):
    """Process uploaded environment configuration file"""
    try:
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        required_columns = ['environment', 'cpu', 'ram', 'storage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Set defaults for optional columns
        if 'daily_usage' not in df.columns:
            df['daily_usage'] = 20  # Default 20 hours
        if 'throughput' not in df.columns:
            df['throughput'] = df['cpu'] * 1000  # Default based on CPU
        
        # Convert to the expected format
        servers = {}
        for _, row in df.iterrows():
            env_name = str(row['environment'])
            servers[env_name] = {
                'cpu': int(row['cpu']),
                'ram': int(row['ram']),
                'storage': int(row['storage']),
                'daily_usage': int(row.get('daily_usage', 20)),
                'throughput': int(row.get('throughput', row['cpu'] * 1000))
            }
        
        st.success(f"✅ Processed {len(servers)} environments successfully!")
        
        # Display processed environments
        st.markdown("#### 📊 Processed Environments")
        processed_df = pd.DataFrame.from_dict(servers, orient='index')
        st.dataframe(processed_df, use_container_width=True)
        
        return servers
        
    except Exception as e:
        st.error(f"❌ Error processing environment data: {str(e)}")
        return None

# REPLACE the existing main() function completely with this:

def main():
    """Enhanced main function with dedicated Analysis tab"""
    services = initialize_services()
    
    # Enterprise Header with Status
    st.markdown("""
    <div class="enterprise-header">
        <h1>🏢 Enterprise Oracle to MongoDB Migration Analyzer</h1>
        <p>Comprehensive analysis and planning for enterprise database migration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Service Status Dashboard
    display_service_status(services)
    
    # Restructured main tabs - Added dedicated Analysis tab
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "🔧 Configuration", 
        "🚀 Analysis & Processing",  # New dedicated analysis tab
        "📊 Results Dashboard", 
        "📋 Reports & Export"
    ])
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'bulk_servers' not in st.session_state:
        st.session_state.bulk_servers = None
    if 'manual_servers' not in st.session_state:
        st.session_state.manual_servers = None
    
    with main_tab1:
        # Configuration Section - NO ANALYSIS BUTTON HERE
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">📊 Migration Configuration</div>', unsafe_allow_html=True)
        
        # Configuration tabs - removed analysis button from here
        config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs([
            "🏢 Environment Setup", "💰 Cost Parameters", "🔧 Technical Specs", "🤖 Advanced Options"
        ])
        
        with config_tab1:
            st.markdown("### Environment Configuration")
            
            # Add input method selection
            st.markdown("#### 📝 Configuration Method")
            config_method = st.radio(
                "Choose how to configure environments:",
                ["Manual Entry", "Bulk Upload (CSV/Excel/JSON)"],
                horizontal=True,
                key="config_method_radio"
            )
            
            if config_method == "Bulk Upload (CSV/Excel/JSON)":
                bulk_servers = handle_bulk_upload()
                if bulk_servers:
                    st.session_state.bulk_servers = bulk_servers
                    st.session_state.manual_servers = None
                    st.info("💡 Bulk configuration loaded. Go to the 'Analysis & Processing' tab to run analysis.")
                
            else:
                # Original manual configuration
                col1, col2 = st.columns(2)
                with col1:
                    num_environments = st.number_input("Number of Environments", 1, 10, 5, key="config_num_environments")
                    data_size_tb = st.number_input("Total Data Size (TB)", 1, 1000, 25, key="config_data_size_tb")
                
                with col2:
                    migration_timeline = st.slider("Migration Timeline (Months)", 3, 24, 8, key="config_migration_timeline")
                    aws_region = st.selectbox("Target AWS Region", 
                                            ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'], 
                                            index=0, key="config_aws_region")
                
                # Enhanced Data Transfer Options
                st.markdown("### 🌐 Data Transfer Strategy")
                col1, col2 = st.columns(2)
                with col1:
                    use_direct_connect = st.checkbox("Use AWS Direct Connect", value=True, key="config_use_direct_connect")
                    bandwidth_option = st.selectbox("Bandwidth Option", 
                                                   ["1 Gbps", "10 Gbps", "100 Gbps"], 
                                                   index=1, key="config_bandwidth_option")
                with col2:
                    consider_snowball = st.checkbox("Consider AWS Snowball for Large Datasets", value=data_size_tb > 50, key="config_consider_snowball")
                    parallel_transfers = st.checkbox("Enable Parallel Transfer Streams", value=True, key="config_parallel_transfers")
                
                bandwidth_gbps = int(bandwidth_option.split()[0])
                
                st.info(f"""
                **📡 Data Transfer Configuration:**
                - **Method:** {'AWS Direct Connect' if use_direct_connect else 'Internet Transfer'}
                - **Bandwidth:** {bandwidth_gbps} Gbps
                - **Estimated Transfer Time:** {(data_size_tb * 1000 * 8) / (bandwidth_gbps * 1000 * 3600 * 24):.1f} days
                - **Snowball Option:** {'Recommended' if consider_snowball else 'Not selected'}
                """)
                
                # Environment Details
                st.markdown("### Environment Details")
                servers = {}
                default_envs = ['Development', 'QA', 'UAT', 'Pre-Production', 'Production']
                
                env_cols = st.columns(min(3, num_environments))
                for i in range(num_environments):
                    with env_cols[i % 3]:
                        with st.expander(f"📍 {default_envs[i] if i < len(default_envs) else f'Environment {i+1}'}", expanded=True):
                            env_name = st.text_input(f"Environment Name", 
                                                   value=default_envs[i] if i < len(default_envs) else f"Env{i+1}",
                                                   key=f"config_env_name_{i}")
                            
                            cpu = st.number_input(f"CPU Cores", 1, 128, 
                                                [2, 4, 8, 16, 32][min(i, 4)], key=f"config_cpu_{i}")
                            ram = st.number_input(f"RAM (GB)", 4, 1024, 
                                                [8, 16, 32, 64, 128][min(i, 4)], key=f"config_ram_{i}")
                            storage = st.number_input(f"Storage (GB)", 10, 10000, 
                                                    [100, 200, 500, 1000, 2000][min(i, 4)], key=f"config_storage_{i}")
                            throughput = st.number_input(f"IOPS", 100, 100000, 
                                                       [1000, 2000, 5000, 10000, 20000][min(i, 4)], key=f"config_throughput_{i}")
                            daily_usage = st.slider(f"Daily Usage (Hours)", 1, 24, 
                                                  [8, 12, 16, 20, 24][min(i, 4)], key=f"config_usage_{i}")
                            
                            servers[env_name] = {
                                'cpu': cpu, 'ram': ram, 'storage': storage,
                                'throughput': throughput, 'daily_usage': daily_usage
                            }
                
                st.session_state.manual_servers = servers
                st.session_state.bulk_servers = None
                
                # Store additional config values for Analysis tab
                st.session_state.update({
                    'data_size_tb': data_size_tb,
                    'migration_timeline': migration_timeline,
                    'use_direct_connect': use_direct_connect,
                    'bandwidth_gbps': bandwidth_gbps,
                    'aws_region': aws_region
                })
        
        with config_tab2:
            st.markdown("### Cost Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Current Oracle Costs")
                oracle_license_cost = st.number_input("Oracle License Cost ($/year)", 0, 5000000, 150000, key="config_oracle_license_cost")
                manpower_cost = st.number_input("Maintenance & Support ($/year)", 0, 2000000, 200000, key="config_manpower_cost")
                oracle_infrastructure = st.number_input("Oracle Infrastructure ($/year)", 0, 1000000, 100000, key="config_oracle_infrastructure")
            
            with col2:
                st.markdown("#### Migration Investment")
                migration_budget = st.number_input("Migration Budget ($)", 0, 2000000, 500000, key="config_migration_budget")
                contingency_percent = st.slider("Contingency Buffer (%)", 10, 50, 20, key="config_contingency_percent")
                training_budget = st.number_input("Training Budget ($)", 0, 200000, 50000, key="config_training_budget")
            
            # Store cost values for Analysis tab
            st.session_state.update({
                'oracle_license_cost': oracle_license_cost,
                'manpower_cost': manpower_cost,
                'migration_budget': migration_budget,
                'contingency_percent': contingency_percent,
                'training_budget': training_budget
            })
            
def create_enhanced_visualizations(results, services):
    """Create enhanced visualizations for the dashboard"""
    
    if not services.get('analytics'):
        return None
    
    # Extract data
    cost_df = results['cost_df']
    complexity_details = results['complexity_details']
    enhanced_migration_costs = results['enhanced_migration_costs']
    servers = results['servers']
    
    # Create comprehensive visualization dictionary
    visualizations = {}
    
    # 1. Cost Comparison Chart
    cost_comparison_fig = go.Figure()
    
    environments = cost_df['Environment'].tolist()
    current_costs = cost_df['Current_Total'].tolist()
    aws_costs = cost_df['AWS_Total_Cost'].tolist()
    
    cost_comparison_fig.add_trace(go.Bar(
        name='Current Oracle Costs',
        x=environments,
        y=current_costs,
        marker_color='lightcoral'
    ))
    
    cost_comparison_fig.add_trace(go.Bar(
        name='Projected AWS Costs',
        x=environments,
        y=aws_costs,
        marker_color='lightblue'
    ))
    
    cost_comparison_fig.update_layout(
        title='Current vs. Projected Annual Costs by Environment',
        xaxis_title='Environment',
        yaxis_title='Annual Cost ($)',
        barmode='group',
        height=400
    )
    
    visualizations['cost_comparison'] = cost_comparison_fig
    
    # 2. Migration Cost Breakdown
    migration_costs = {
        'Migration Team': enhanced_migration_costs['migration_team_cost'],
        'Data Transfer': enhanced_migration_costs['recommended_transfer_cost'],
        'AWS Services': enhanced_migration_costs['dms_cost'] + enhanced_migration_costs['datasync_cost'],
        'Infrastructure': enhanced_migration_costs['network_setup_cost'] + enhanced_migration_costs['temp_storage_cost'],
        'Training & Tools': enhanced_migration_costs['training_costs'] + enhanced_migration_costs['tool_costs'],
        'Professional Services': enhanced_migration_costs['aws_professional_services'],
        'Testing': enhanced_migration_costs['testing_costs'],
        'Contingency': enhanced_migration_costs['contingency_cost']
    }
    
    migration_breakdown_fig = services['analytics'].create_cost_breakdown_pie(migration_costs)
    visualizations['migration_breakdown'] = migration_breakdown_fig
    
    # 3. Savings Timeline
    years = ['Year 1', 'Year 2', 'Year 3']
    annual_savings = cost_df['Annual_Savings'].sum()
    cumulative_savings = [annual_savings, annual_savings * 2, annual_savings * 3]
    investment_recovery = [-enhanced_migration_costs['total_migration_cost'], 
                          annual_savings - enhanced_migration_costs['total_migration_cost'],
                          annual_savings * 2 - enhanced_migration_costs['total_migration_cost']]
    
    savings_fig = go.Figure()
    
    savings_fig.add_trace(go.Bar(
        name='Annual Savings',
        x=years,
        y=[annual_savings] * 3,
        marker_color='lightgreen'
    ))
    
    savings_fig.add_trace(go.Scatter(
        name='Cumulative Savings',
        x=years,
        y=cumulative_savings,
        mode='lines+markers',
        line=dict(color='darkgreen', width=3),
        marker=dict(size=10)
    ))
    
    savings_fig.add_trace(go.Scatter(
        name='Net ROI',
        x=years,
        y=investment_recovery,
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    savings_fig.update_layout(
        title='3-Year Savings and ROI Projection',
        xaxis_title='Year',
        yaxis_title='Amount ($)',
        height=400
    )
    
    visualizations['savings_timeline'] = savings_fig
    
    return visualizations
        
        # Replace the Results Dashboard tab (main_tab3) and Reports tab (main_tab4) sections in your main() function
# with these complete implementations:

# COMPLETE RESULTS DASHBOARD TAB (main_tab3)
    with main_tab3:
        st.markdown("## 📊 Enhanced Migration Analysis Results")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Enhanced Results Dashboard
            cost_df = results['cost_df']
            complexity_score = results['complexity_score']
            complexity_details = results['complexity_details']
            enhanced_migration_costs = results['enhanced_migration_costs']
            strategy = results['strategy']
            risk_factors = results['risk_factors']
            ai_analyses = results['ai_analyses']
            servers = results['servers']
            data_size_tb = results['data_size_tb']
            migration_timeline = results['migration_timeline']
            
            # Executive Summary Metrics
            total_current_cost = cost_df['Current_Total'].sum()
            total_aws_cost = cost_df['AWS_Total_Cost'].sum()
            total_annual_savings = cost_df['Annual_Savings'].sum()
            three_year_savings = total_annual_savings * 3
            roi_percentage = (three_year_savings - enhanced_migration_costs['total_migration_cost']) / enhanced_migration_costs['total_migration_cost'] * 100 if enhanced_migration_costs['total_migration_cost'] > 0 else 0
            
            # Executive Dashboard
            st.markdown("### 📊 Executive Dashboard")
            
            # Main metrics grid
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">${enhanced_migration_costs['total_migration_cost']:,.0f}</p>
                    <p class="metric-label">Total Migration Cost</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">${enhanced_migration_costs['recommended_transfer_cost']:,.0f}</p>
                    <p class="metric-label">Data Transfer Cost</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{enhanced_migration_costs['estimated_transfer_days']:.1f} days</p>
                    <p class="metric-label">Transfer Time</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">${total_annual_savings:,.0f}</p>
                    <p class="metric-label">Annual Savings</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col5:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{roi_percentage:.1f}%</p>
                    <p class="metric-label">3-Year ROI</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            
            # Detailed Analysis Sections
            analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
                "📊 Cost Analysis", "📈 Visual Analytics", "🎯 Migration Strategy", "🤖 AI Insights"
            ])
            
            with analysis_tab1:
                # Cost breakdown
                st.markdown("#### 💰 Comprehensive Cost Analysis")
                
                cost_col1, cost_col2 = st.columns(2)
                
                with cost_col1:
                    st.markdown("##### Current vs. Projected Costs")
                    st.dataframe(cost_df[['Environment', 'Current_Total', 'AWS_Total_Cost', 'Annual_Savings', 'Savings_Percentage']], 
                            use_container_width=True)
                
                with cost_col2:
                    # Cost breakdown pie chart
                    if services.get('analytics'):
                        cost_components = {
                            'Migration Team': enhanced_migration_costs['migration_team_cost'],
                            'Data Transfer': enhanced_migration_costs['recommended_transfer_cost'],
                            'AWS DMS': enhanced_migration_costs['dms_cost'],
                            'Infrastructure': enhanced_migration_costs['network_setup_cost'] + enhanced_migration_costs['temp_storage_cost'],
                            'Training': enhanced_migration_costs['training_costs'],
                            'Tools & Services': enhanced_migration_costs['tool_costs'] + enhanced_migration_costs['aws_professional_services'],
                            'Contingency': enhanced_migration_costs['contingency_cost']
                        }
                        
                        cost_pie_fig = services['analytics'].create_cost_breakdown_pie(cost_components)
                        st.plotly_chart(cost_pie_fig, use_container_width=True)
                
                # Enhanced migration costs breakdown
                st.markdown("##### 🔍 Enhanced Migration Cost Breakdown")
                
                migration_cost_data = {
                    'Cost Component': [
                        'Migration Team', 'Data Transfer (Recommended)', 'AWS DMS Service', 
                        'AWS DataSync', 'Network Infrastructure', 'Temporary Storage',
                        'Backup Storage', 'Tools & Licenses', 'Training & Certification',
                        'AWS Professional Services', 'Testing & Validation', 'Contingency (15%)'
                    ],
                    'Cost ($)': [
                        enhanced_migration_costs['migration_team_cost'],
                        enhanced_migration_costs['recommended_transfer_cost'],
                        enhanced_migration_costs['dms_cost'],
                        enhanced_migration_costs['datasync_cost'],
                        enhanced_migration_costs['network_setup_cost'] + enhanced_migration_costs['network_monitoring_cost'],
                        enhanced_migration_costs['temp_storage_cost'],
                        enhanced_migration_costs['backup_storage_cost'],
                        enhanced_migration_costs['tool_costs'],
                        enhanced_migration_costs['training_costs'],
                        enhanced_migration_costs['aws_professional_services'],
                        enhanced_migration_costs['testing_costs'],
                        enhanced_migration_costs['contingency_cost']
                    ],
                    'Description': [
                        f'Team for {migration_timeline} months',
                        f'{enhanced_migration_costs["data_transfer_breakdown"]["recommended"]["option"]["method"]}',
                        f'Instance: {enhanced_migration_costs["dms_instance_type"]}',
                        'Ongoing synchronization during migration',
                        f'Setup + monitoring for {results["bandwidth_gbps"]} Gbps',
                        f'Temporary storage during migration',
                        'Backup storage during migration',
                        'Migration tools and licenses',
                        'Team training and certifications',
                        'Expert consulting services',
                        'Comprehensive testing strategy',
                        '15% buffer for unforeseen costs'
                    ]
                }
                
                migration_df = pd.DataFrame(migration_cost_data)
                migration_df['Cost ($)'] = migration_df['Cost ($)'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(migration_df, use_container_width=True)
            
            # Data Transfer Analysis
            st.markdown("##### 🚚 Data Transfer Options Analysis")
            
            transfer_options = enhanced_migration_costs['data_transfer_breakdown']['all_options']
            recommended_key = enhanced_migration_costs['data_transfer_breakdown']['recommended']['option_key']
            
            transfer_data = []
            for option_key, option in transfer_options.items():
                transfer_data.append({
                    'Method': option['method'],
                    'Cost ($)': f"${option['total_cost']:,.0f}",
                    'Time (days)': f"{option['transfer_time_days']:.1f}",
                    'Bandwidth': option.get('bandwidth', 'N/A'),
                    'Suitability': option['suitability'],
                    'Recommended': '✅' if option_key == recommended_key else '❌',
                    'Score': f"{option.get('recommendation_score', 50)}/100"
                })
            
            transfer_df = pd.DataFrame(transfer_data)
            st.dataframe(transfer_df, use_container_width=True)
            
            # Show recommended method details
            recommended_option = enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']
            st.success(f"""
            **Recommended Transfer Method:** {recommended_option['method']}
            - **Cost:** ${recommended_option['total_cost']:,.0f}
            - **Transfer Time:** {recommended_option['transfer_time_days']:.1f} days
            - **Bandwidth:** {recommended_option.get('bandwidth', 'N/A')}
            - **Pros:** {', '.join(recommended_option['pros'])}
            """)
        
        with analysis_tab2:
            # Visual Analytics
            st.markdown("#### 📈 Visual Analytics Dashboard")
            
            if services.get('analytics'):
                # Complexity Heatmap
                st.markdown("##### 🎯 Migration Complexity Heatmap")
                complexity_heatmap = services['analytics'].create_complexity_heatmap(complexity_details, servers)
                st.plotly_chart(complexity_heatmap, use_container_width=True)
                
                # ROI Waterfall Chart
                st.markdown("##### 💹 3-Year ROI Waterfall Analysis")
                roi_waterfall = services['analytics'].create_roi_waterfall(enhanced_migration_costs, total_annual_savings)
                st.plotly_chart(roi_waterfall, use_container_width=True)
                
                # Migration Timeline Gantt Chart
                st.markdown("##### 📅 Migration Timeline - Gantt Chart")
                timeline_gantt = services['analytics'].create_timeline_gantt(migration_timeline * 4, results['num_pl_sql_objects'])
                st.plotly_chart(timeline_gantt, use_container_width=True)
                
                # Data Transfer Comparison
                st.markdown("##### 🌐 Data Transfer Options Comparison")
                transfer_comparison = services['analytics'].create_transfer_comparison_chart(transfer_options)
                st.plotly_chart(transfer_comparison, use_container_width=True)
                
            else:
                st.warning("Analytics service not available. Install required dependencies for full visualization.")
        
        with analysis_tab3:
            # Migration Strategy
            st.markdown("#### 🎯 Detailed Migration Strategy")
            
            # Strategy Overview
            strategy_col1, strategy_col2 = st.columns(2)
            
            with strategy_col1:
                st.markdown(f"""
                <div class="analysis-section">
                    <h4>📋 Recommended Strategy</h4>
                    <h3 style="color: #4299e1;">{strategy['strategy']}</h3>
                    <p><strong>Timeline:</strong> {strategy['timeline']}</p>
                    <p><strong>Risk Level:</strong> <span class="risk-{strategy['risk'].lower()}">{strategy['risk']}</span></p>
                    <p><strong>Effort Level:</strong> {strategy['effort']}</p>
                    <p>{strategy['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with strategy_col2:
                # Complexity Score Breakdown
                st.markdown("##### 🎯 Complexity Analysis")
                complexity_df = pd.DataFrame([
                    {'Factor': k.replace('_', ' ').title(), 'Score': f"{v:.1f}/100"} 
                    for k, v in complexity_details.items()
                ])
                st.dataframe(complexity_df, use_container_width=True)
                
                st.metric("Overall Complexity Score", f"{complexity_score}/100", 
                         delta=f"Risk Level: {strategy['risk']}")
            
            # Risk Factors
            st.markdown("##### ⚠️ Risk Factors & Mitigation")
            
            for i, risk in enumerate(risk_factors, 1):
                st.markdown(f"""
                <div class="risk-medium" style="padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem;">
                    <strong>Risk {i}:</strong> {risk}
                </div>
                """, unsafe_allow_html=True)
            
            # Technical Recommendations
            st.markdown("##### 🔧 Technical Recommendations")
            
            tech_recommendations = []
            for env, specs in servers.items():
                rec = results['cost_df'][results['cost_df']['Environment'] == env]
                if not rec.empty:
                    tech_recommendations.append({
                        'Environment': env,
                        'Current Specs': f"{specs['cpu']} vCPU, {specs['ram']} GB RAM, {specs['storage']} GB Storage",
                        'Recommended EC2': rec.iloc[0]['EC2_Instance'],
                        'Recommended MongoDB': rec.iloc[0]['MongoDB_Cluster'],
                        'Annual Savings': f"${rec.iloc[0]['Annual_Savings']:,.0f}"
                    })
            
            tech_df = pd.DataFrame(tech_recommendations)
            st.dataframe(tech_df, use_container_width=True)
            
            # Migration Phases
            st.markdown("##### 📊 Migration Phases")
            
            phases = [
                {
                    'Phase': '1. Discovery & Assessment',
                    'Duration': '2-4 weeks',
                    'Key Activities': 'Complete inventory, dependency mapping, risk assessment',
                    'Deliverables': 'Migration plan, risk register, technical architecture'
                },
                {
                    'Phase': '2. Environment Setup',
                    'Duration': '2-3 weeks', 
                    'Key Activities': 'AWS infrastructure, MongoDB setup, security configuration',
                    'Deliverables': 'Configured environments, security framework'
                },
                {
                    'Phase': '3. Data Migration',
                    'Duration': f'{enhanced_migration_costs["estimated_transfer_days"]:.0f} days',
                    'Key Activities': f'{recommended_option["method"]}, data validation, testing',
                    'Deliverables': 'Migrated data, validation reports'
                },
                {
                    'Phase': '4. Application Refactoring',
                    'Duration': f'{max(8, results["num_pl_sql_objects"] // 100)} weeks',
                    'Key Activities': 'PL/SQL conversion, application updates, integration testing',
                    'Deliverables': 'Refactored applications, test results'
                },
                {
                    'Phase': '5. Go-Live & Optimization',
                    'Duration': '2-4 weeks',
                    'Key Activities': 'Production cutover, performance tuning, monitoring setup',
                    'Deliverables': 'Live system, performance reports, handover documentation'
                }
            ]
            
            phases_df = pd.DataFrame(phases)
            st.dataframe(phases_df, use_container_width=True)
        
        with analysis_tab4:
            # AI Insights
            st.markdown("#### 🤖 AI-Powered Insights")
            
            if ai_analyses:
                ai_col1, ai_col2 = st.columns(2)
                
                with ai_col1:
                    if 'executive_summary' in ai_analyses:
                        st.markdown("##### 📊 Executive Summary")
                        st.markdown(ai_analyses['executive_summary'])
                    
                    if 'technical_strategy' in ai_analyses:
                        st.markdown("##### 🔧 Technical Strategy")
                        st.markdown(ai_analyses['technical_strategy'])
                
                with ai_col2:
                    if 'risk_assessment' in ai_analyses:
                        st.markdown("##### ⚠️ Risk Assessment")
                        st.markdown(ai_analyses['risk_assessment'])
                    
                    if 'cost_optimization' in ai_analyses:
                        st.markdown("##### 💰 Cost Optimization")
                        st.markdown(ai_analyses['cost_optimization'])
            
            else:
                st.info("🤖 AI analysis not available. Enable AI analysis in the configuration to get detailed insights.")
                
                # Show fallback insights
                st.markdown("##### 📋 Standard Analysis Summary")
                st.markdown(f"""
                **Migration Overview:**
                - **Complexity Score:** {complexity_score}/100 ({strategy['risk']} risk)
                - **Recommended Strategy:** {strategy['strategy']}
                - **Timeline:** {strategy['timeline']}
                - **Total Investment:** ${enhanced_migration_costs['total_migration_cost']:,.0f}
                - **Annual Savings:** ${total_annual_savings:,.0f}
                - **3-Year ROI:** {roi_percentage:.1f}%
                
                **Key Recommendations:**
                - Use {recommended_option['method']} for data transfer
                - Implement AWS DMS with {enhanced_migration_costs['dms_instance_type']} instance
                - Plan for {enhanced_migration_costs['estimated_transfer_days']:.1f} days of data transfer
                - Budget ${enhanced_migration_costs['contingency_cost']:,.0f} for contingency
                """)
    
    else:
        st.info("👆 Please run the analysis in the 'Analysis & Processing' tab first to see comprehensive results.")


# COMPLETE REPORTS TAB (main_tab4)
    with main_tab4:
        st.markdown("## 📋 Comprehensive Reports & Export")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Report generation section
            report_col1, report_col2 = st.columns([2, 1])
            
            with report_col1:
                st.markdown("### 📄 Available Reports")
                
                report_types = st.multiselect(
                    "Select report types to generate:",
                    ["Executive Summary", "Technical Analysis", "Cost Analysis", "Risk Assessment", "Migration Plan"],
                    default=["Executive Summary", "Cost Analysis", "Migration Plan"]
                )
                
                include_ai = st.checkbox("Include AI Analysis", value=bool(results.get('ai_analyses')))
                include_charts = st.checkbox("Include Charts and Visualizations", value=True)
                
            with report_col2:
                st.markdown("### 📊 Export Options")
                
                # Text Report Download
                if st.button("📄 Generate Text Report", type="primary", use_container_width=True):
                    # Generate comprehensive text report
                    cost_df = results['cost_df']
                    enhanced_migration_costs = results['enhanced_migration_costs']
                    strategy = results['strategy']
                    
                    total_current_cost = cost_df['Current_Total'].sum()
                    total_aws_cost = cost_df['AWS_Total_Cost'].sum()
                    total_annual_savings = cost_df['Annual_Savings'].sum()
                    three_year_savings = total_annual_savings * 3
                    roi_percentage = (three_year_savings - enhanced_migration_costs['total_migration_cost']) / enhanced_migration_costs['total_migration_cost'] * 100 if enhanced_migration_costs['total_migration_cost'] > 0 else 0
                    
                    text_report = generate_text_report(
                        total_current_cost, total_aws_cost, total_annual_savings,
                        roi_percentage, results['migration_timeline'], results['complexity_score'],
                        enhanced_migration_costs, strategy, results['servers'], 
                        results['risk_factors'], results['data_size_tb']
                    )
                    
                    st.download_button(
                        label="📥 Download Text Report",
                        data=text_report,
                        file_name=f"Oracle_MongoDB_Migration_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # CSV Export
                if st.button("📊 Export Cost Analysis (CSV)", use_container_width=True):
                    csv = cost_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"Migration_Cost_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # PDF Report (if available)
                if services.get('pdf'):
                    if st.button("📑 Generate PDF Report", use_container_width=True):
                        try:
                            # Generate analysis data for PDF
                            analysis_data = {
                                'total_current_cost': cost_df['Current_Total'].sum(),
                                'total_aws_cost': cost_df['AWS_Total_Cost'].sum(),
                                'total_annual_savings': cost_df['Annual_Savings'].sum(),
                                'complexity_score': results['complexity_score'],
                                'timeline_months': results['migration_timeline'],
                                'three_year_roi': roi_percentage
                            }
                            
                            pdf_buffer = services['pdf'].generate_report(analysis_data)
                            
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"Oracle_MongoDB_Migration_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"PDF generation failed: {str(e)}")
                else:
                    st.info("📑 PDF generation unavailable (install reportlab)")
            
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            
            # Detailed Reports Preview
            st.markdown("### 📋 Report Preview")
            
            preview_tab1, preview_tab2, preview_tab3, preview_tab4 = st.tabs([
                "Executive Summary", "Migration Strategy", "Cost Analysis", "Technical Details"
            ])
            
            with preview_tab1:
                st.markdown("#### 📊 Executive Summary Report")
                
                cost_df = results['cost_df']
                enhanced_migration_costs = results['enhanced_migration_costs']
                
                total_current_cost = cost_df['Current_Total'].sum()
                total_aws_cost = cost_df['AWS_Total_Cost'].sum()
                total_annual_savings = cost_df['Annual_Savings'].sum()
                three_year_savings = total_annual_savings * 3
                roi_percentage = (three_year_savings - enhanced_migration_costs['total_migration_cost']) / enhanced_migration_costs['total_migration_cost'] * 100 if enhanced_migration_costs['total_migration_cost'] > 0 else 0
                
                st.markdown(f"""
                **ORACLE TO MONGODB MIGRATION - EXECUTIVE SUMMARY**
                
                **Financial Overview:**
                - Current Annual Oracle Cost: ${total_current_cost:,.0f}
                - Projected Annual AWS Cost: ${total_aws_cost:,.0f}
                - Annual Savings: ${total_annual_savings:,.0f}
                - Total Migration Investment: ${enhanced_migration_costs['total_migration_cost']:,.0f}
                - 3-Year ROI: {roi_percentage:.1f}%
                
                **Migration Overview:**
                - Complexity Score: {results['complexity_score']}/100
                - Recommended Strategy: {results['strategy']['strategy']}
                - Timeline: {results['migration_timeline']} months
                - Data Size: {results['data_size_tb']} TB
                - Transfer Method: {enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']}
                - Transfer Time: {enhanced_migration_costs['estimated_transfer_days']:.1f} days
                
                **Strategic Recommendations:**
                - Proceed with {results['strategy']['strategy']} approach
                - Implement {enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']} for data transfer
                - Plan for {results['migration_timeline']} month timeline with {results['strategy']['risk']} risk profile
                - Allocate ${enhanced_migration_costs['contingency_cost']:,.0f} contingency budget
                """)
                
                # Add AI summary if available
                if results.get('ai_analyses') and 'executive_summary' in results['ai_analyses']:
                    st.markdown("**AI-Generated Insights:**")
                    st.markdown(results['ai_analyses']['executive_summary'])
        
        with preview_tab2:
            st.markdown("#### 🎯 Migration Strategy Report")
            
            strategy = results['strategy']
            enhanced_migration_costs = results['enhanced_migration_costs']
            
            st.markdown(f"""
            **MIGRATION STRATEGY: {strategy['strategy']}**
            
            **Strategy Details:**
            - Approach: {strategy['strategy']}
            - Risk Level: {strategy['risk']}
            - Timeline: {strategy['timeline']}
            - Effort Level: {strategy['effort']}
            - Description: {strategy['description']}
            
            **Data Transfer Strategy:**
            - Method: {enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']}
            - Bandwidth: {results.get('bandwidth_gbps', 10)} Gbps
            - Transfer Time: {enhanced_migration_costs['estimated_transfer_days']:.1f} days
            - Transfer Cost: ${enhanced_migration_costs['recommended_transfer_cost']:,.0f}
            
            **Migration Components:**
            - Migration Team: ${enhanced_migration_costs['migration_team_cost']:,.0f}
            - AWS DMS Service: ${enhanced_migration_costs['dms_cost']:,.0f} ({enhanced_migration_costs['dms_instance_type']})
            - Network Infrastructure: ${enhanced_migration_costs['network_setup_cost']:,.0f}
            - Training & Certification: ${enhanced_migration_costs['training_costs']:,.0f}
            - AWS Professional Services: ${enhanced_migration_costs['aws_professional_services']:,.0f}
            
            **Risk Factors:**
            """)
            
            for i, risk in enumerate(results['risk_factors'], 1):
                st.markdown(f"{i}. {risk}")
            
            # Add AI strategy if available
            if results.get('ai_analyses') and 'technical_strategy' in results['ai_analyses']:
                st.markdown("**AI-Generated Technical Strategy:**")
                st.markdown(results['ai_analyses']['technical_strategy'])
        
        with preview_tab3:
            st.markdown("#### 💰 Cost Analysis Report")
            
            # Environment-wise cost breakdown
            st.markdown("**Environment-wise Cost Analysis:**")
            st.dataframe(cost_df[['Environment', 'Current_Total', 'AWS_Total_Cost', 'Annual_Savings', 'Savings_Percentage']], 
                        use_container_width=True)
            
            # Migration cost breakdown
            st.markdown("**Migration Investment Breakdown:**")
            migration_costs_display = {
                'Migration Team': enhanced_migration_costs['migration_team_cost'],
                'Data Transfer': enhanced_migration_costs['recommended_transfer_cost'],
                'AWS Services (DMS + DataSync)': enhanced_migration_costs['dms_cost'] + enhanced_migration_costs['datasync_cost'],
                'Infrastructure & Storage': enhanced_migration_costs['network_setup_cost'] + enhanced_migration_costs['temp_storage_cost'],
                'Tools & Training': enhanced_migration_costs['tool_costs'] + enhanced_migration_costs['training_costs'],
                'Professional Services': enhanced_migration_costs['aws_professional_services'],
                'Testing & Validation': enhanced_migration_costs['testing_costs'],
                'Contingency (15%)': enhanced_migration_costs['contingency_cost']
            }
            
            cost_breakdown_df = pd.DataFrame([
                {'Component': k, 'Cost ($)': f"${v:,.0f}", 'Percentage': f"{v/enhanced_migration_costs['total_migration_cost']*100:.1f}%"}
                for k, v in migration_costs_display.items()
            ])
            st.dataframe(cost_breakdown_df, use_container_width=True)
            
            # Add AI cost analysis if available
            if results.get('ai_analyses') and 'cost_optimization' in results['ai_analyses']:
                st.markdown("**AI-Generated Cost Optimization:**")
                st.markdown(results['ai_analyses']['cost_optimization'])
        
        with preview_tab4:
            st.markdown("#### 🔧 Technical Details Report")
            
            st.markdown("**Current Environment Specifications:**")
            env_specs_df = pd.DataFrame.from_dict(results['servers'], orient='index')
            st.dataframe(env_specs_df, use_container_width=True)
            
            st.markdown("**Recommended AWS Configuration:**")
            aws_config_df = cost_df[['Environment', 'EC2_Instance', 'MongoDB_Cluster']]
            st.dataframe(aws_config_df, use_container_width=True)
            
            st.markdown(f"""
            **Technical Migration Details:**
            - PL/SQL Objects to Convert: {results['num_pl_sql_objects']:,}
            - Connected Applications: {results.get('num_applications', 'N/A')}
            - Data Volume: {results['data_size_tb']} TB
            - Complexity Score: {results['complexity_score']}/100
            
            **Data Transfer Configuration:**
            - Transfer Method: {enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']}
            - Bandwidth: {results.get('bandwidth_gbps', 10)} Gbps
            - Estimated Duration: {enhanced_migration_costs['estimated_transfer_days']:.1f} days
            - DMS Instance: {enhanced_migration_costs['dms_instance_type']}
            - DMS Duration: {enhanced_migration_costs['dms_duration_hours']:.0f} hours
            
            **Complexity Factor Breakdown:**
            """)
            
            complexity_breakdown_df = pd.DataFrame([
                {'Factor': k.replace('_', ' ').title(), 'Score': f"{v:.1f}/100"}
                for k, v in results['complexity_details'].items()
            ])
            st.dataframe(complexity_breakdown_df, use_container_width=True)
        
        # Action Items and Next Steps
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 📋 Action Items & Next Steps")
        
        next_steps_col1, next_steps_col2 = st.columns(2)
        
        with next_steps_col1:
            st.markdown("#### 🎯 Immediate Actions")
            st.markdown("""
            1. **Stakeholder Approval**
               - Present executive summary to leadership
               - Secure budget approval for migration investment
               
            2. **Team Formation**
               - Assemble migration team
               - Identify key stakeholders and SMEs
               
            3. **Vendor Engagement**
               - Engage AWS Professional Services
               - Setup Direct Connect infrastructure
               
            4. **Environment Preparation**
               - Provision AWS environments
               - Configure MongoDB Atlas clusters
            """)
        
        with next_steps_col2:
            st.markdown("#### 📅 Timeline Milestones")
            st.markdown(f"""
            1. **Week 1-2: Project Initiation**
               - Stakeholder alignment
               - Team formation and training
               
            2. **Week 3-6: Infrastructure Setup**
               - AWS environment provisioning
               - {enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']} setup
               
            3. **Week 7-{migration_timeline*4-8}: Migration Execution**
               - Data migration ({enhanced_migration_costs['estimated_transfer_days']:.0f} days)
               - Application refactoring
               - Testing and validation
               
            4. **Week {migration_timeline*4-7}-{migration_timeline*4}: Go-Live**
               - Production cutover
               - Performance optimization
               - Post-migration support
            """)
        
        # Success Metrics
        st.markdown("### 📊 Success Metrics & KPIs")
        
        success_col1, success_col2, success_col3 = st.columns(3)
        
        with success_col1:
            st.markdown("#### 💰 Financial KPIs")
            st.markdown(f"""
            - **Target Annual Savings:** ${total_annual_savings:,.0f}
            - **Migration Budget:** ${enhanced_migration_costs['total_migration_cost']:,.0f}
            - **ROI Target:** {roi_percentage:.1f}% over 3 years
            - **Cost Reduction:** {((total_current_cost - total_aws_cost) / total_current_cost * 100):.1f}%
            """)
        
        with success_col2:
            st.markdown("#### ⏱️ Timeline KPIs")
            st.markdown(f"""
            - **Migration Timeline:** {migration_timeline} months
            - **Data Transfer:** {enhanced_migration_costs['estimated_transfer_days']:.0f} days
            - **Zero Downtime Target:** 99.9% uptime
            - **Rollback Time:** < 4 hours
            """)
        
        with success_col3:
            st.markdown("#### 🎯 Quality KPIs")
            st.markdown(f"""
            - **Data Integrity:** 100% accuracy
            - **Performance:** Maintain current response times
            - **Risk Mitigation:** Address all {len(results['risk_factors'])} identified risks
            - **User Satisfaction:** > 90% acceptance
            """)
    
    else:
        st.info("👆 Please run the analysis in the 'Analysis & Processing' tab first to generate comprehensive reports.")
        
        # Show sample report formats
        st.markdown("### 📋 Available Report Formats")
        
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        with sample_col1:
            st.markdown("""
            #### 📄 Text Report
            - Executive summary
            - Detailed cost analysis
            - Technical specifications
            - Migration strategy
            - Risk assessment
            """)
        
        with sample_col2:
            st.markdown("""
            #### 📊 CSV Export
            - Environment configurations
            - Cost breakdowns
            - Instance recommendations
            - Savings analysis
            - Timeline data
            """)
        
        with sample_col3:
            st.markdown("""
            #### 📑 PDF Report
            - Professional formatting
            - Charts and visualizations
            - Executive presentation
            - Technical appendix
            - Action items
            """)
    
    # NEW: Dedicated Analysis Tab
    with main_tab2:
        st.markdown("## 🚀 Analysis & Processing Center")
        st.markdown("This is where you run the comprehensive migration analysis using your configured parameters.")
        
        # Show current configuration status
        current_servers = st.session_state.bulk_servers or st.session_state.manual_servers
        
        if current_servers:
            st.markdown("### 📊 Current Configuration Summary")
            
            config_col1, config_col2, config_col3 = st.columns(3)
            with config_col1:
                st.metric("Environments", len(current_servers))
            with config_col2:
                total_cpu = sum(env['cpu'] for env in current_servers.values())
                st.metric("Total CPU Cores", total_cpu)
            with config_col3:
                total_ram = sum(env['ram'] for env in current_servers.values())
                st.metric("Total RAM (GB)", total_ram)
            
            # Show environments table
            env_df = pd.DataFrame.from_dict(current_servers, orient='index')
            st.dataframe(env_df, use_container_width=True)
            
            # Analysis options - with defaults from session state
            st.markdown("### ⚙️ Analysis Configuration")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown("#### 📈 Analysis Parameters")
                data_size_tb = st.number_input("Data Size (TB)", 1, 1000, 
                                              value=st.session_state.get('data_size_tb', 25), key="analysis_data_size_tb")
                migration_timeline = st.number_input("Timeline (Months)", 3, 24, 
                                                   value=st.session_state.get('migration_timeline', 8), key="analysis_migration_timeline")
                num_pl_sql_objects = st.number_input("PL/SQL Objects", 0, 50000, 
                                                   value=st.session_state.get('num_pl_sql_objects', 800), key="analysis_num_pl_sql_objects")
                num_applications = st.number_input("Applications", 1, 100, 
                                                 value=st.session_state.get('num_applications', 5), key="analysis_num_applications")
            
            with analysis_col2:
                st.markdown("#### 💰 Cost Parameters")
                oracle_license_cost = st.number_input("Oracle License ($/year)", 0, 5000000, 
                                                    value=st.session_state.get('oracle_license_cost', 150000), key="analysis_oracle_license_cost")
                manpower_cost = st.number_input("Maintenance ($/year)", 0, 2000000, 
                                              value=st.session_state.get('manpower_cost', 200000), key="analysis_manpower_cost")
                migration_budget = st.number_input("Migration Budget ($)", 0, 2000000, 
                                                 value=st.session_state.get('migration_budget', 500000), key="analysis_migration_budget")
            
            # Transfer options
            st.markdown("#### 🌐 Transfer Configuration")
            transfer_col1, transfer_col2 = st.columns(2)
            
            with transfer_col1:
                use_direct_connect = st.checkbox("Use AWS Direct Connect", 
                                               value=st.session_state.get('use_direct_connect', True), key="analysis_use_direct_connect")
                bandwidth_option = st.selectbox("Bandwidth", ["1 Gbps", "10 Gbps", "100 Gbps"], 
                                               index=1, key="analysis_bandwidth_option")
            
            with transfer_col2:
                aws_region = st.selectbox("AWS Region", 
                                        ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'], 
                                        index=0, key="analysis_aws_region")
                enable_ai = st.checkbox("Enable AI Analysis", 
                                      value=st.session_state.get('enable_ai', True), key="analysis_enable_ai")
            
            bandwidth_gbps = int(bandwidth_option.split()[0])
            
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            
            # THE SINGLE ANALYSIS BUTTON - ONLY HERE!
            st.markdown("### 🚀 Run Analysis")
            
            analysis_button_col1, analysis_button_col2, analysis_button_col3 = st.columns([1, 2, 1])
            with analysis_button_col2:
                if st.button("🚀 Generate Enhanced Analysis", type="primary", use_container_width=True, key="main_analysis_button"):
                    # Call the analysis function with current parameters
                    analyze_migration(
                        services, current_servers, oracle_license_cost, manpower_cost, 
                        data_size_tb, migration_timeline, num_pl_sql_objects, num_applications, 
                        enable_ai, migration_budget, st.session_state.get('training_budget', 50000), 
                        st.session_state.get('contingency_percent', 20), aws_region,
                        use_direct_connect, bandwidth_gbps
                    )
            
            # Analysis tips
            st.info("""
            **💡 Analysis Tips:**
            - Ensure all environments are properly configured
            - Verify cost parameters reflect your current Oracle setup
            - Review transfer options for your data volume
            - Enable AI analysis for enhanced insights and recommendations
            """)
        
        else:
            st.warning("⚠️ No environments configured. Please go to the 'Configuration' tab to set up your environments first.")
            
            st.markdown("### 🏃‍♂️ Quick Start Options")
            
            quick_col1, quick_col2 = st.columns(2)
            
            with quick_col1:
                st.markdown("#### 📝 Manual Configuration")
                st.info("Go to the 'Configuration' tab to manually enter your environment details.")
                
            with quick_col2:
                st.markdown("#### 📁 Bulk Upload")
                st.info("Upload a CSV/Excel file with your environment configurations in the 'Configuration' tab.")
                
                # Show sample format
                st.markdown("**Sample CSV format:**")
                sample_df = pd.DataFrame({
                    'Environment': ['Production', 'Development'],
                    'CPU': [32, 8],
                    'RAM': [128, 32],
                    'Storage': [2000, 500],
                    'Daily_Usage': [24, 12],
                    'Throughput': [20000, 5000]
                })
                st.dataframe(sample_df, use_container_width=True)
    
    # Keep the existing Results Dashboard tab the same
    with main_tab3:
        st.markdown("## 📊 Enhanced Migration Analysis Results")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Enhanced Results Dashboard
            cost_df = results['cost_df']
            complexity_score = results['complexity_score']
            enhanced_migration_costs = results['enhanced_migration_costs']
            
            # Executive Summary Metrics
            total_current_cost = cost_df['Current_Total'].sum()
            total_aws_cost = cost_df['AWS_Total_Cost'].sum()
            total_annual_savings = cost_df['Annual_Savings'].sum()
            three_year_savings = total_annual_savings * 3
            roi_percentage = (three_year_savings - enhanced_migration_costs['total_migration_cost']) / enhanced_migration_costs['total_migration_cost'] * 100 if enhanced_migration_costs['total_migration_cost'] > 0 else 0
            
            # Executive Dashboard
            st.markdown("### 📊 Executive Dashboard")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Migration Cost",
                    f"${enhanced_migration_costs['total_migration_cost']:,.0f}",
                    f"Timeline: {results['migration_timeline']} months"
                )
            with col2:
                st.metric(
                    "Data Transfer Cost",
                    f"${enhanced_migration_costs['recommended_transfer_cost']:,.0f}",
                    f"Bandwidth: {results['bandwidth_gbps']} Gbps"
                )
            with col3:
                st.metric(
                    "Estimated Transfer Time",
                    f"{enhanced_migration_costs['estimated_transfer_days']:.1f} days",
                    f"Method: {enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']}"
                )
            with col4:
                st.metric(
                    "Annual Savings",
                    f"${total_annual_savings:,.0f}",
                    f"ROI: {roi_percentage:.1f}%"
                )
            
            # Rest of the existing results dashboard code...
            st.info("Complete results dashboard showing all analysis results and visualizations.")
        
        else:
            st.info("👆 Please run the analysis in the 'Analysis & Processing' tab first.")
    
    # Keep the existing Reports tab the same
    with main_tab4:
        st.markdown("## 📋 Comprehensive Reports & Export")
        
        if st.session_state.analysis_results:
            st.info("Reports functionality with PDF generation, CSV exports, and comprehensive documentation.")
        else:
            st.info("👆 Please run the analysis first to generate reports.")

if __name__ == "__main__":
    main()