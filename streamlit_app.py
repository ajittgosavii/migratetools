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

# Main Application with Reports Tab
def main():
    """Enhanced main function with dedicated Reports tab"""
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
    
    # Main tabs including Reports
    main_tab1, main_tab2, main_tab3 = st.tabs([
        "🔧 Configuration & Analysis", 
        "📊 Results Dashboard", 
        "📋 Reports & Export"
    ])
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    with main_tab1:
        # Configuration Section
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">📊 Migration Configuration</div>', unsafe_allow_html=True)
        
        # Configuration tabs
        config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs([
            "🏢 Environment Setup", "💰 Cost Parameters", "🔧 Technical Specs", "🤖 Advanced Options"
        ])
        
        with config_tab1:
            st.markdown("### Environment Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                num_environments = st.number_input("Number of Environments", 1, 10, 5)
                data_size_tb = st.number_input("Total Data Size (TB)", 1, 1000, 25)
            
            with col2:
                migration_timeline = st.slider("Migration Timeline (Months)", 3, 24, 8)
                aws_region = st.selectbox("Target AWS Region", 
                                        ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'], 
                                        index=0)
            
            # Enhanced Data Transfer Options
            st.markdown("### 🌐 Data Transfer Strategy")
            col1, col2 = st.columns(2)
            with col1:
                use_direct_connect = st.checkbox("Use AWS Direct Connect", value=True)
                bandwidth_option = st.selectbox("Bandwidth Option", 
                                               ["1 Gbps", "10 Gbps", "100 Gbps"], 
                                               index=1)
            with col2:
                consider_snowball = st.checkbox("Consider AWS Snowball for Large Datasets", value=data_size_tb > 50)
                parallel_transfers = st.checkbox("Enable Parallel Transfer Streams", value=True)
            
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
                                               key=f"env_name_{i}")
                        
                        cpu = st.number_input(f"CPU Cores", 1, 128, 
                                            [2, 4, 8, 16, 32][min(i, 4)], key=f"cpu_{i}")
                        ram = st.number_input(f"RAM (GB)", 4, 1024, 
                                            [8, 16, 32, 64, 128][min(i, 4)], key=f"ram_{i}")
                        storage = st.number_input(f"Storage (GB)", 10, 10000, 
                                                [100, 200, 500, 1000, 2000][min(i, 4)], key=f"storage_{i}")
                        throughput = st.number_input(f"IOPS", 100, 100000, 
                                                   [1000, 2000, 5000, 10000, 20000][min(i, 4)], key=f"throughput_{i}")
                        daily_usage = st.slider(f"Daily Usage (Hours)", 1, 24, 
                                              [8, 12, 16, 20, 24][min(i, 4)], key=f"usage_{i}")
                        
                        servers[env_name] = {
                            'cpu': cpu, 'ram': ram, 'storage': storage,
                            'throughput': throughput, 'daily_usage': daily_usage
                        }
        
        with config_tab2:
            st.markdown("### Cost Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Current Oracle Costs")
                oracle_license_cost = st.number_input("Oracle License Cost ($/year)", 0, 5000000, 150000)
                manpower_cost = st.number_input("Maintenance & Support ($/year)", 0, 2000000, 200000)
                oracle_infrastructure = st.number_input("Oracle Infrastructure ($/year)", 0, 1000000, 100000)
            
            with col2:
                st.markdown("#### Migration Investment")
                migration_budget = st.number_input("Migration Budget ($)", 0, 2000000, 500000)
                contingency_percent = st.slider("Contingency Buffer (%)", 10, 50, 20)
                training_budget = st.number_input("Training Budget ($)", 0, 200000, 50000)
        
        with config_tab3:
            st.markdown("### Technical Specifications")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Application Architecture")
                num_pl_sql_objects = st.number_input("PL/SQL Objects Count", 0, 50000, 800)
                num_applications = st.number_input("Connected Applications", 1, 100, 5)
                integration_endpoints = st.number_input("API/Integration Endpoints", 0, 500, 25)
            
            with col2:
                st.markdown("#### Operational Requirements")
                backup_retention = st.selectbox("Backup Retention (Days)", [7, 14, 30, 90, 365], index=2)
                high_availability = st.selectbox("High Availability", ["Standard", "Multi-AZ", "Multi-Region"], index=1)
                compliance_requirements = st.multiselect("Compliance Requirements", 
                                                        ["SOX", "HIPAA", "PCI-DSS", "GDPR", "ISO-27001"], 
                                                        default=["SOX"])
        
        with config_tab4:
            st.markdown("### Advanced Options")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### AI Analysis")
                enable_ai = st.checkbox("Enable AI-Powered Analysis", True)
                if enable_ai and ANTHROPIC_AVAILABLE and 'anthropic_api_key' not in st.session_state:
                    api_key = st.text_input("Anthropic API Key", type="password", 
                                           help="Get your API key from console.anthropic.com")
                    if api_key:
                        st.session_state.anthropic_api_key = api_key
                        # Reinitialize AI service with new key
                        services['ai'] = EnterpriseAIService()
            
            with col2:
                st.markdown("#### AWS Integration")
                use_real_pricing = st.checkbox("Fetch Real-time AWS Pricing", value=AWS_API_AVAILABLE)
                if use_real_pricing and not AWS_API_AVAILABLE:
                    st.warning("AWS SDK not available. Install boto3 and configure credentials.")
                
                optimization_level = st.selectbox("Cost Optimization Level", 
                                                ["Conservative", "Balanced", "Aggressive"], index=1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis Button
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Enhanced Analysis", type="primary", use_container_width=True):
                analyze_migration(services, servers, oracle_license_cost, manpower_cost, data_size_tb, 
                                migration_timeline, num_pl_sql_objects, num_applications, enable_ai,
                                migration_budget, training_budget, contingency_percent, aws_region,
                                use_direct_connect, bandwidth_gbps)
    
    with main_tab2:
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
            
            # Data Transfer Options Comparison
            st.markdown("### 🌐 Data Transfer Options Analysis")
            
            transfer_options = enhanced_migration_costs['data_transfer_breakdown']['all_options']
            recommended_key = enhanced_migration_costs['data_transfer_breakdown']['recommended']['option_key']
            
            # Create comparison table
            comparison_data = []
            for option_key, option in transfer_options.items():
                is_recommended = option_key == recommended_key
                comparison_data.append({
                    'Method': f"{'✅ ' if is_recommended else ''}{option['method']}",
                    'Bandwidth': option.get('bandwidth', 'N/A'),
                    'Transfer Time (Days)': f"{option['transfer_time_days']:.1f}",
                    'Total Cost': f"${option['total_cost']:,.0f}",
                    'Suitability': option['suitability'],
                    'Score': option.get('recommendation_score', 'N/A')
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Create transfer options visualization
            if results.get('analytics'):
                transfer_chart = results['analytics'].create_transfer_comparison_chart(transfer_options)
                st.plotly_chart(transfer_chart, use_container_width=True)
            
            # Enhanced Cost Breakdown
            st.markdown("### 💰 Enhanced Migration Cost Breakdown")
            
            cost_breakdown = {
                'Component': [
                    'Migration Team',
                    'Data Transfer (Recommended)',
                    'AWS DMS Service',
                    'AWS DataSync',
                    'Network Infrastructure',
                    'Temporary Storage',
                    'Backup Storage',
                    'Tools & Licensing',
                    'Training & Certification',
                    'AWS Professional Services',
                    'Testing & Validation',
                    'Contingency (15%)',
                    'Total Investment'
                ],
                'Cost': [
                    f"${enhanced_migration_costs['migration_team_cost']:,.0f}",
                    f"${enhanced_migration_costs['recommended_transfer_cost']:,.0f}",
                    f"${enhanced_migration_costs['dms_cost']:,.0f}",
                    f"${enhanced_migration_costs['datasync_cost']:,.0f}",
                    f"${enhanced_migration_costs['network_setup_cost'] + enhanced_migration_costs['network_monitoring_cost']:,.0f}",
                    f"${enhanced_migration_costs['temp_storage_cost']:,.0f}",
                    f"${enhanced_migration_costs['backup_storage_cost']:,.0f}",
                    f"${enhanced_migration_costs['tool_costs']:,.0f}",
                    f"${enhanced_migration_costs['training_costs']:,.0f}",
                    f"${enhanced_migration_costs['aws_professional_services']:,.0f}",
                    f"${enhanced_migration_costs['testing_costs']:,.0f}",
                    f"${enhanced_migration_costs['contingency_cost']:,.0f}",
                    f"${enhanced_migration_costs['total_migration_cost']:,.0f}"
                ],
                'Details': [
                    f"Project team for {results['migration_timeline']} months",
                    f"{enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']}",
                    f"{enhanced_migration_costs['dms_instance_type']} for {enhanced_migration_costs['dms_duration_hours']:.0f} hours",
                    "Ongoing synchronization during migration",
                    f"Network setup and {results['migration_timeline']} months monitoring",
                    f"Staging storage for {results['data_size_tb']} TB",
                    f"Backup storage during migration period",
                    "Enterprise migration tools and licenses",
                    "Team training and AWS certifications",
                    "AWS expert consulting services",
                    "Comprehensive testing and validation",
                    "Risk mitigation buffer",
                    "Complete migration investment"
                ]
            }
            
            breakdown_df = pd.DataFrame(cost_breakdown)
            st.dataframe(breakdown_df, use_container_width=True)
            
            # Cost comparison chart
            st.markdown("### 📊 Cost Comparison by Environment")
            environments = cost_df['Environment']
            current_costs = cost_df['Current_Total']
            aws_costs = cost_df['AWS_Total_Cost']
            
            cost_comparison_fig = go.Figure()
            
            cost_comparison_fig.add_trace(go.Bar(
                name='Current Oracle Costs',
                x=environments,
                y=current_costs,
                marker_color='#FF6B6B',
                text=[f'${cost:,.0f}' for cost in current_costs],
                textposition='auto'
            ))
            
            cost_comparison_fig.add_trace(go.Bar(
                name='Projected AWS Costs',
                x=environments,
                y=aws_costs,
                marker_color='#4ECDC4',
                text=[f'${cost:,.0f}' for cost in aws_costs],
                textposition='auto'
            ))
            
            cost_comparison_fig.update_layout(
                title='Annual Cost Comparison by Environment',
                xaxis_title='Environment',
                yaxis_title='Annual Cost ($)',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(cost_comparison_fig, use_container_width=True)
            
            # Data Transfer Deep Dive
            st.markdown("### 🌐 Data Transfer Deep Dive")
            
            transfer_analysis = enhanced_migration_costs['data_transfer_breakdown']
            recommended = transfer_analysis['recommended']['option']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Recommended Solution")
                st.success(f"**{recommended['method']}**")
                st.info(f"""
                **Configuration:**
                - Bandwidth: {recommended.get('bandwidth', 'N/A')}
                - Transfer Time: {recommended['transfer_time_days']:.1f} days
                - Total Cost: ${recommended['total_cost']:,.0f}
                
                **Why Recommended:**
                - {recommended['suitability']}
                - {', '.join(recommended['pros'][:2])}
                """)
            
            with col2:
                st.markdown("#### AWS Services Breakdown")
                
                aws_services_data = {
                    'Service': ['DMS Instance', 'DataSync', 'Network Setup', 'Monitoring'],
                    'Cost': [
                        f"${enhanced_migration_costs['dms_cost']:,.0f}",
                        f"${enhanced_migration_costs['datasync_cost']:,.0f}",
                        f"${enhanced_migration_costs['network_setup_cost']:,.0f}",
                        f"${enhanced_migration_costs['network_monitoring_cost']:,.0f}"
                    ],
                    'Purpose': [
                        f"Database migration ({enhanced_migration_costs['dms_instance_type']})",
                        "Ongoing synchronization",
                        "Infrastructure setup",
                        f"Monitoring for {results['migration_timeline']} months"
                    ]
                }
                
                aws_services_df = pd.DataFrame(aws_services_data)
                st.dataframe(aws_services_df, use_container_width=True)
            
            # Enhanced info box for data transfer
            st.info(f"""
            **📡 Data Transfer Summary:**
            
            • **Volume:** {results['data_size_tb']} TB ({results['data_size_tb'] * 1000:,.0f} GB)
            • **Method:** {recommended['method']}
            • **Bandwidth:** {results['bandwidth_gbps']} Gbps
            • **Transfer Time:** {enhanced_migration_costs['estimated_transfer_days']:.1f} days
            • **Total Cost:** ${enhanced_migration_costs['recommended_transfer_cost']:,.0f}
            • **DMS Duration:** {enhanced_migration_costs['dms_duration_hours']:.0f} hours
            • **Instance Type:** {enhanced_migration_costs['dms_instance_type']}
            """)
            
            # Risk Assessment
            if results['risk_factors']:
                st.markdown("### 🎯 Risk Assessment")
                for i, risk in enumerate(results['risk_factors']):
                    risk_level = "high" if i < 2 else "medium" if i < 4 else "low"
                    st.markdown(f"""
                    <div class="metric-card risk-{risk_level}">
                        <strong>Risk {i+1}:</strong> {risk}
                    </div>
                    """, unsafe_allow_html=True)
            
            # AI Analysis Display
            if results.get('ai_analyses'):
                st.markdown("### 🤖 AI Analysis Insights")
                
                for analysis_type, content in results['ai_analyses'].items():
                    with st.expander(f"{analysis_type.replace('_', ' ').title()}", expanded=False):
                        st.markdown(content)
        
        else:
            st.info("👆 Please run the analysis in the Configuration & Analysis tab first.")
    
    # Add this enhanced PDF functionality to the Reports tab section

# Replace the existing Reports tab content (main_tab3) with this enhanced version:

with main_tab3:
    st.markdown("## 📋 Comprehensive Reports & Export")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        enhanced_migration_costs = results['enhanced_migration_costs']
        cost_df = results['cost_df']
        
        # Report Generation Options
        st.markdown("### 📄 Available Reports")
        
        report_col1, report_col2, report_col3, report_col4 = st.columns(4)
        
        with report_col1:
            st.markdown("#### 📊 Executive Summary")
            
            total_current_cost = cost_df['Current_Total'].sum()
            total_aws_cost = cost_df['AWS_Total_Cost'].sum()
            total_annual_savings = cost_df['Annual_Savings'].sum()
            
            executive_summary = f"""
**Oracle to MongoDB Migration Analysis**
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Migration Investment: ${enhanced_migration_costs['total_migration_cost']:,.0f}
Data Volume: {results['data_size_tb']} TB
Timeline: {results['migration_timeline']} months
Annual Savings: ${total_annual_savings:,.0f}

DATA TRANSFER STRATEGY
=====================
Recommended Method: {enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']}
Transfer Time: {enhanced_migration_costs['estimated_transfer_days']:.1f} days
Bandwidth: {results['bandwidth_gbps']} Gbps
Transfer Cost: ${enhanced_migration_costs['recommended_transfer_cost']:,.0f}

AWS SERVICES
============
DMS Instance: {enhanced_migration_costs['dms_instance_type']}
DMS Duration: {enhanced_migration_costs['dms_duration_hours']:.0f} hours
DMS Cost: ${enhanced_migration_costs['dms_cost']:,.0f}
DataSync Cost: ${enhanced_migration_costs['datasync_cost']:,.0f}

KEY RECOMMENDATIONS
==================
• Implement {enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']}
• Use {enhanced_migration_costs['dms_instance_type']} for DMS
• Budget ${enhanced_migration_costs['total_migration_cost']:,.0f} for complete migration
• Plan for {enhanced_migration_costs['estimated_transfer_days']:.1f} days data transfer
            """
            
            st.download_button(
                "📊 Executive Summary",
                executive_summary,
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with report_col2:
            st.markdown("#### 🔧 Technical Report")
            
            # Generate comprehensive technical report
            technical_report = generate_text_report(
                total_current_cost, total_aws_cost, total_annual_savings,
                ((total_annual_savings * 3 - enhanced_migration_costs['total_migration_cost']) / enhanced_migration_costs['total_migration_cost'] * 100),
                results['migration_timeline'], results['complexity_score'],
                enhanced_migration_costs, results['strategy'], results['servers'], 
                results['risk_factors'], results['data_size_tb']
            )
            
            st.download_button(
                "🔧 Technical Report",
                technical_report,
                file_name=f"technical_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with report_col3:
            st.markdown("#### ⚙️ Configuration")
            
            # Comprehensive configuration export
            config_export = {
                'metadata': {
                    'generated': datetime.now().isoformat(),
                    'version': '2.0',
                    'analysis_type': 'Enhanced Oracle to MongoDB Migration'
                },
                'configuration': {
                    'data_size_tb': results['data_size_tb'],
                    'migration_timeline_months': results['migration_timeline'],
                    'use_direct_connect': results['use_direct_connect'],
                    'bandwidth_gbps': results['bandwidth_gbps'],
                    'environments': results['servers']
                },
                'migration_analysis': {
                    'total_cost': enhanced_migration_costs['total_migration_cost'],
                    'transfer_analysis': enhanced_migration_costs['data_transfer_breakdown'],
                    'recommended_method': enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method'],
                    'transfer_time_days': enhanced_migration_costs['estimated_transfer_days'],
                    'complexity_score': results['complexity_score']
                },
                'aws_services': {
                    'dms_instance': enhanced_migration_costs['dms_instance_type'],
                    'dms_duration_hours': enhanced_migration_costs['dms_duration_hours'],
                    'dms_cost': enhanced_migration_costs['dms_cost'],
                    'datasync_cost': enhanced_migration_costs['datasync_cost']
                },
                'cost_analysis': cost_df.to_dict('records')
            }
            
            config_json = json.dumps(config_export, indent=2, default=str)
            
            st.download_button(
                "⚙️ Full Configuration",
                config_json,
                file_name=f"migration_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with report_col4:
            st.markdown("#### 📑 Professional PDF")
            
            # Enhanced PDF generation function
            def generate_comprehensive_pdf():
                """Generate a comprehensive PDF report"""
                try:
                    if not PDF_AVAILABLE:
                        return None, "PDF generation requires reportlab library. Please install: pip install reportlab"
                    
                    from reportlab.lib.pagesizes import letter, A4
                    from reportlab.lib import colors
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
                    from reportlab.lib.units import inch
                    from reportlab.graphics.shapes import Drawing
                    from reportlab.graphics.charts.piecharts import Pie
                    from reportlab.graphics.charts.barcharts import VerticalBarChart
                    
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                                          topMargin=72, bottomMargin=18)
                    
                    # Custom styles
                    styles = getSampleStyleSheet()
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=24,
                        spaceAfter=30,
                        textColor=colors.HexColor('#2D3748'),
                        alignment=1  # Center alignment
                    )
                    
                    heading_style = ParagraphStyle(
                        'CustomHeading',
                        parent=styles['Heading2'],
                        fontSize=16,
                        spaceAfter=12,
                        textColor=colors.HexColor('#4299E1')
                    )
                    
                    body_style = ParagraphStyle(
                        'CustomBody',
                        parent=styles['Normal'],
                        fontSize=11,
                        spaceAfter=8,
                        textColor=colors.HexColor('#2D3748')
                    )
                    
                    story = []
                    
                    # Title Page
                    story.append(Paragraph("Oracle to MongoDB Migration Analysis", title_style))
                    story.append(Spacer(1, 20))
                    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", body_style))
                    story.append(Spacer(1, 50))
                    
                    # Executive Summary
                    story.append(Paragraph("Executive Summary", heading_style))
                    executive_data = [
                        ['Metric', 'Current State', 'Target State', 'Impact'],
                        ['Annual Cost', f"${total_current_cost:,.0f}", 
                         f"${total_aws_cost:,.0f}", 
                         f"${total_annual_savings:,.0f} savings"],
                        ['Data Volume', f"{results['data_size_tb']} TB", 
                         f"MongoDB Atlas", f"Enhanced scalability"],
                        ['Migration Investment', '', 
                         f"${enhanced_migration_costs['total_migration_cost']:,.0f}", 
                         f"{results['migration_timeline']} months"],
                        ['Complexity Score', '', f"{results['complexity_score']}/100", 
                         f"{results['strategy']['risk']} risk level"],
                        ['Data Transfer', '', 
                         f"{enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']['method']}", 
                         f"{enhanced_migration_costs['estimated_transfer_days']:.1f} days"]
                    ]
                    
                    exec_table = Table(executive_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                    exec_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4299E1')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    story.append(exec_table)
                    story.append(Spacer(1, 30))
                    
                    # Data Transfer Strategy
                    story.append(Paragraph("Data Transfer Strategy", heading_style))
                    recommended_option = enhanced_migration_costs['data_transfer_breakdown']['recommended']['option']
                    
                    transfer_content = f"""
                    <b>Recommended Method:</b> {recommended_option['method']}<br/>
                    <b>Data Volume:</b> {results['data_size_tb']} TB ({results['data_size_tb'] * 1000:,.0f} GB)<br/>
                    <b>Bandwidth:</b> {results['bandwidth_gbps']} Gbps<br/>
                    <b>Transfer Time:</b> {enhanced_migration_costs['estimated_transfer_days']:.1f} days<br/>
                    <b>Transfer Cost:</b> ${enhanced_migration_costs['recommended_transfer_cost']:,.0f}<br/>
                    <b>Suitability:</b> {recommended_option['suitability']}<br/><br/>
                    
                    <b>Key Benefits:</b><br/>
                    • {recommended_option['pros'][0] if recommended_option['pros'] else 'Optimized for your data volume'}<br/>
                    • {recommended_option['pros'][1] if len(recommended_option['pros']) > 1 else 'Enterprise-grade reliability'}<br/>
                    • Dedicated bandwidth ensuring predictable transfer times<br/>
                    • Built-in monitoring and validation capabilities
                    """
                    
                    story.append(Paragraph(transfer_content, body_style))
                    story.append(Spacer(1, 20))
                    
                    # AWS Services Configuration
                    story.append(Paragraph("AWS Services Configuration", heading_style))
                    
                    aws_services_data = [
                        ['Service', 'Configuration', 'Duration', 'Cost', 'Purpose'],
                        ['Database Migration Service', 
                         enhanced_migration_costs['dms_instance_type'], 
                         f"{enhanced_migration_costs['dms_duration_hours']:.0f} hours",
                         f"${enhanced_migration_costs['dms_cost']:,.0f}",
                         'Primary database migration'],
                        ['DataSync', 
                         'Ongoing synchronization', 
                         f"{results['migration_timeline']} months",
                         f"${enhanced_migration_costs['datasync_cost']:,.0f}",
                         'Data synchronization'],
                        ['Network Infrastructure', 
                         f"{results['bandwidth_gbps']} Gbps Direct Connect", 
                         f"{results['migration_timeline']} months",
                         f"${enhanced_migration_costs['network_setup_cost']:,.0f}",
                         'Dedicated connectivity'],
                        ['Professional Services', 
                         'AWS Expert Consultation', 
                         'As needed',
                         f"${enhanced_migration_costs['aws_professional_services']:,.0f}",
                         'Expert guidance']
                    ]
                    
                    aws_table = Table(aws_services_data, colWidths=[1.2*inch, 1.2*inch, 1*inch, 1*inch, 1.6*inch])
                    aws_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#48BB78')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    story.append(aws_table)
                    story.append(Spacer(1, 30))
                    
                    # Cost Breakdown
                    story.append(Paragraph("Detailed Cost Breakdown", heading_style))
                    
                    cost_breakdown_data = [
                        ['Cost Component', 'Amount', 'Percentage', 'Details'],
                        ['Migration Team', 
                         f"${enhanced_migration_costs['migration_team_cost']:,.0f}",
                         f"{(enhanced_migration_costs['migration_team_cost']/enhanced_migration_costs['total_migration_cost']*100):.1f}%",
                         f"Project team for {results['migration_timeline']} months"],
                        ['Data Transfer', 
                         f"${enhanced_migration_costs['recommended_transfer_cost']:,.0f}",
                         f"{(enhanced_migration_costs['recommended_transfer_cost']/enhanced_migration_costs['total_migration_cost']*100):.1f}%",
                         f"{recommended_option['method']}"],
                        ['AWS DMS Service', 
                         f"${enhanced_migration_costs['dms_cost']:,.0f}",
                         f"{(enhanced_migration_costs['dms_cost']/enhanced_migration_costs['total_migration_cost']*100):.1f}%",
                         f"{enhanced_migration_costs['dms_instance_type']} instance"],
                        ['Infrastructure & Network', 
                         f"${enhanced_migration_costs['network_setup_cost'] + enhanced_migration_costs['network_monitoring_cost']:,.0f}",
                         f"{((enhanced_migration_costs['network_setup_cost'] + enhanced_migration_costs['network_monitoring_cost'])/enhanced_migration_costs['total_migration_cost']*100):.1f}%",
                         f"Network setup and monitoring"],
                        ['Storage & Backup', 
                         f"${enhanced_migration_costs['temp_storage_cost'] + enhanced_migration_costs['backup_storage_cost']:,.0f}",
                         f"{((enhanced_migration_costs['temp_storage_cost'] + enhanced_migration_costs['backup_storage_cost'])/enhanced_migration_costs['total_migration_cost']*100):.1f}%",
                         f"Temporary and backup storage"],
                        ['Tools & Training', 
                         f"${enhanced_migration_costs['tool_costs'] + enhanced_migration_costs['training_costs']:,.0f}",
                         f"{((enhanced_migration_costs['tool_costs'] + enhanced_migration_costs['training_costs'])/enhanced_migration_costs['total_migration_cost']*100):.1f}%",
                         f"Licenses and team training"],
                        ['Contingency', 
                         f"${enhanced_migration_costs['contingency_cost']:,.0f}",
                         f"{(enhanced_migration_costs['contingency_cost']/enhanced_migration_costs['total_migration_cost']*100):.1f}%",
                         f"Risk mitigation buffer"],
                        ['Total Investment', 
                         f"${enhanced_migration_costs['total_migration_cost']:,.0f}",
                         '100.0%',
                         f"Complete migration cost"]
                    ]
                    
                    cost_table = Table(cost_breakdown_data, colWidths=[1.8*inch, 1.2*inch, 1*inch, 2*inch])
                    cost_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ED8936')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#FED7D7')),
                        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                        ('BACKGROUND', (0, 1), (-1, -2), colors.lightblue),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    story.append(cost_table)
                    story.append(PageBreak())
                    
                    # Environment Analysis
                    story.append(Paragraph("Environment Analysis", heading_style))
                    
                    env_data = [['Environment', 'CPU (Cores)', 'RAM (GB)', 'Storage (GB)', 'Daily Usage (Hours)', 'AWS Recommendation']]
                    for env_name, specs in results['servers'].items():
                        # Get recommendations
                        rec = recommend_instances({env_name: specs})[env_name]
                        env_data.append([
                            env_name,
                            str(specs['cpu']),
                            str(specs['ram']),
                            str(specs['storage']),
                            str(specs['daily_usage']),
                            f"EC2: {rec['ec2']}, MongoDB: {rec['mongodb']}"
                        ])
                    
                    env_table = Table(env_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch, 1.4*inch])
                    env_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9F7AEA')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    story.append(env_table)
                    story.append(Spacer(1, 20))
                    
                    # Risk Assessment
                    if results['risk_factors']:
                        story.append(Paragraph("Risk Assessment & Mitigation", heading_style))
                        
                        risk_content = "<b>Identified Risks and Mitigation Strategies:</b><br/><br/>"
                        for i, risk in enumerate(results['risk_factors'], 1):
                            risk_level = "HIGH" if i <= 2 else "MEDIUM" if i <= 4 else "LOW"
                            risk_content += f"<b>{i}. [{risk_level} RISK]</b> {risk}<br/><br/>"
                        
                        story.append(Paragraph(risk_content, body_style))
                        story.append(Spacer(1, 20))
                    
                    # Implementation Timeline
                    story.append(Paragraph("Implementation Timeline", heading_style))
                    
                    timeline_content = f"""
                    <b>Total Timeline:</b> {results['migration_timeline']} months<br/>
                    <b>Data Transfer Duration:</b> {enhanced_migration_costs['estimated_transfer_days']:.1f} days<br/>
                    <b>Strategy:</b> {results['strategy']['strategy']}<br/>
                    <b>Risk Level:</b> {results['strategy']['risk']}<br/><br/>
                    
                    <b>Key Phases:</b><br/>
                    1. <b>Planning & Assessment (Weeks 1-4):</b> Detailed analysis and planning<br/>
                    2. <b>Infrastructure Setup (Weeks 3-9):</b> AWS environment and network configuration<br/>
                    3. <b>Data Migration (Weeks 8-{8 + enhanced_migration_costs['estimated_transfer_days']/7:.0f}):</b> Primary data transfer using {recommended_option['method']}<br/>
                    4. <b>Testing & Validation (Weeks {results['migration_timeline']*4-8:.0f}-{results['migration_timeline']*4-4:.0f}):</b> Comprehensive testing and validation<br/>
                    5. <b>Go-Live & Support (Final 2-3 weeks):</b> Production cutover and stabilization
                    """
                    
                    story.append(Paragraph(timeline_content, body_style))
                    story.append(Spacer(1, 30))
                    
                    # Next Steps
                    story.append(Paragraph("Next Steps & Recommendations", heading_style))
                    
                    next_steps_content = f"""
                    <b>Immediate Actions (Next 30 days):</b><br/>
                    1. Stakeholder approval for ${enhanced_migration_costs['total_migration_cost']:,.0f} budget<br/>
                    2. AWS Direct Connect setup planning for {results['bandwidth_gbps']} Gbps<br/>
                    3. Migration team formation and training initiation<br/>
                    4. {enhanced_migration_costs['dms_instance_type']} DMS instance sizing confirmation<br/><br/>
                    
                    <b>Short-term (60-90 days):</b><br/>
                    1. Complete infrastructure setup and network connectivity<br/>
                    2. Begin data profiling and migration testing<br/>
                    3. Establish monitoring and backup procedures<br/>
                    4. Finalize {recommended_option['method']} configuration<br/><br/>
                    
                    <b>Long-term Success Factors:</b><br/>
                    • Comprehensive testing with production-like data<br/>
                    • Phased migration approach to minimize business impact<br/>
                    • Continuous monitoring and optimization post-migration<br/>
                    • Regular checkpoints and progress reviews
                    """
                    
                    story.append(Paragraph(next_steps_content, body_style))
                    
                    # Build PDF
                    doc.build(story)
                    buffer.seek(0)
                    return buffer, None
                    
                except Exception as e:
                    return None, f"PDF generation failed: {str(e)}"
            
            # PDF download button
            if st.button("📑 Generate Professional PDF", use_container_width=True, type="primary"):
                with st.spinner("Generating comprehensive PDF report..."):
                    pdf_buffer, error = generate_comprehensive_pdf()
                    
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download Professional PDF Report",
                            data=pdf_buffer,
                            file_name=f"oracle_mongodb_migration_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.success("✅ PDF report generated successfully!")
                    else:
                        st.error(f"❌ {error}")
                        st.info("💡 Install reportlab for PDF generation: `pip install reportlab`")
            
            # PDF preview info
            st.info("""
            **📑 Professional PDF Report includes:**
            • Executive summary with key metrics
            • Detailed data transfer strategy
            • AWS services configuration
            • Comprehensive cost breakdown
            • Environment analysis & recommendations
            • Risk assessment & mitigation
            • Implementation timeline
            • Next steps & action items
            """)
        
        # Enhanced transfer analysis section (rest of the content remains the same)
        st.markdown("### 🌐 Data Transfer Detailed Analysis")
        
        transfer_options = enhanced_migration_costs['data_transfer_breakdown']['all_options']
        
        # Create detailed comparison cards
        for option_key, option in transfer_options.items():
            is_recommended = option_key == enhanced_migration_costs['data_transfer_breakdown']['recommended']['option_key']
            card_class = "recommended-option" if is_recommended else "transfer-option-card"
            
            st.markdown(f"""
            <div class="transfer-option-card {card_class}">
                <h4>{'🏆 ' if is_recommended else ''}
                {option['method']} {'(Recommended)' if is_recommended else ''}</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
                    <div><strong>Bandwidth:</strong> {option.get('bandwidth', 'N/A')}</div>
                    <div><strong>Transfer Time:</strong> {option['transfer_time_days']:.1f} days</div>
                    <div><strong>Total Cost:</strong> ${option['total_cost']:,.0f}</div>
                    <div><strong>Suitability:</strong> {option['suitability']}</div>
                </div>
                <div style="margin: 1rem 0;">
                    <strong>Pros:</strong> {', '.join(option['pros'])}
                </div>
                <div>
                    <strong>Cons:</strong> {', '.join(option['cons'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Cost breakdown visualization
        st.markdown("### 💰 Migration Cost Visualization")
        
        # Create pie chart for migration costs
        cost_components = {
            'Migration Team': enhanced_migration_costs['migration_team_cost'],
            'Data Transfer': enhanced_migration_costs['recommended_transfer_cost'],
            'AWS Services': enhanced_migration_costs['dms_cost'] + enhanced_migration_costs['datasync_cost'],
            'Infrastructure': enhanced_migration_costs['network_setup_cost'] + enhanced_migration_costs['network_monitoring_cost'],
            'Storage': enhanced_migration_costs['temp_storage_cost'] + enhanced_migration_costs['backup_storage_cost'],
            'Tools & Training': enhanced_migration_costs['tool_costs'] + enhanced_migration_costs['training_costs'],
            'Professional Services': enhanced_migration_costs['aws_professional_services'],
            'Testing': enhanced_migration_costs['testing_costs'],
            'Contingency': enhanced_migration_costs['contingency_cost']
        }
        
        cost_pie_fig = go.Figure(data=[go.Pie(
            labels=list(cost_components.keys()),
            values=list(cost_components.values()),
            hole=0.3,
            textinfo='label+percent',
            texttemplate='%{label}<br>%{percent}',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        cost_pie_fig.update_layout(
            title="Migration Cost Breakdown",
            height=600,
            annotations=[dict(text=f'Total<br>${enhanced_migration_costs["total_migration_cost"]:,.0f}', 
                            x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        st.plotly_chart(cost_pie_fig, use_container_width=True)
        
        # Timeline visualization
        st.markdown("### 📅 Migration Timeline")
        
        timeline_phases = [
            {'Phase': 'Planning & Assessment', 'Start': 0, 'Duration': 4, 'Color': '#FF9999'},
            {'Phase': 'Infrastructure Setup', 'Start': 3, 'Duration': 6, 'Color': '#66B2FF'},
            {'Phase': 'Data Migration', 'Start': 8, 'Duration': enhanced_migration_costs['estimated_transfer_days']/7, 'Color': '#99FF99'},
            {'Phase': 'Testing & Validation', 'Start': 8 + enhanced_migration_costs['estimated_transfer_days']/7, 'Duration': 4, 'Color': '#FFCC99'},
            {'Phase': 'Go-Live & Support', 'Start': results['migration_timeline'] - 2, 'Duration': 3, 'Color': '#FF99CC'}
        ]
        
        timeline_fig = go.Figure()
        
        for phase in timeline_phases:
            timeline_fig.add_trace(go.Bar(
                x=[phase['Duration']],
                y=[phase['Phase']],
                orientation='h',
                name=phase['Phase'],
                marker=dict(color=phase['Color']),
                text=f"{phase['Duration']:.1f} weeks",
                textposition='inside',
                base=phase['Start']
            ))
        
        timeline_fig.update_layout(
            title='Migration Timeline Overview',
            xaxis_title='Weeks',
            yaxis_title='Migration Phases',
            height=400,
            showlegend=False,
            xaxis=dict(range=[0, results['migration_timeline'] + 2])
        )
        
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # CSV export for detailed data
        st.markdown("### 📊 Detailed Data Export")
        
        csv_col1, csv_col2 = st.columns(2)
        
        with csv_col1:
            # Cost analysis CSV
            csv_data = cost_df.to_csv(index=False)
            st.download_button(
                "📊 Cost Analysis (CSV)",
                csv_data,
                file_name=f"cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with csv_col2:
            # Transfer options CSV
            transfer_df = pd.DataFrame([
                {
                    'Method': option['method'],
                    'Bandwidth': option.get('bandwidth', 'N/A'),
                    'Transfer_Time_Days': option['transfer_time_days'],
                    'Total_Cost': option['total_cost'],
                    'Suitability': option['suitability'],
                    'Recommendation_Score': option.get('recommendation_score', 'N/A')
                }
                for option in transfer_options.values()
            ])
            
            transfer_csv = transfer_df.to_csv(index=False)
            st.download_button(
                "🌐 Transfer Options (CSV)",
                transfer_csv,
                file_name=f"transfer_options_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.info("👆 Please run the analysis first to generate reports.")
        
        # Show sample report templates
        st.markdown("### 📋 Available Report Templates")
        
        sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
        
        with sample_col1:
            st.markdown("""
            #### 📊 Executive Summary
            - Migration overview and ROI
            - Data transfer strategy
            - Cost-benefit analysis  
            - Timeline and milestones
            - Key recommendations
            """)
        
        with sample_col2:
            st.markdown("""
            #### 🔧 Technical Deep Dive
            - Infrastructure analysis
            - Enhanced data transfer options
            - AWS services configuration
            - DMS and DataSync setup
            - Performance projections
            """)
        
        with sample_col3:
            st.markdown("""
            #### ⚙️ Configuration Export
            - Complete settings backup
            - Environment specifications
            - Cost parameters
            - Technical configurations
            - Analysis metadata
            """)
        
        with sample_col4:
            st.markdown("""
            #### 📑 Professional PDF
            - Comprehensive analysis report
            - Executive-ready presentation
            - Detailed cost breakdowns
            - Visual charts and graphs
            - Implementation roadmap
            """)

if __name__ == "__main__":
    main()