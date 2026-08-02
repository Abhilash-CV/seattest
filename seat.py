import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from itertools import product

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Define the seat matrix based on categories
SEAT_MATRIX = {
    'SM': 50, 'EW': 10, 'EZ': 9, 'MU': 8, 'SC': 8, 
    'BH': 3, 'LA': 3, 'DV': 2, 'VK': 2, 'ST': 2, 
    'KN': 1, 'BX': 1, 'KU': 1
}

# Page configuration
st.set_page_config(
    page_title="Seat Allocation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2c8bcb;
        color: white;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        color: #155724;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SEAT ALLOCATOR CLASS
# ============================================================================

class SeatAllocator:
    """
    Seat allocation using Hamilton rounding and biproportional methods
    """
    
    def __init__(self, college_data):
        """
        Initialize with college data
        
        Args:
            college_data: DataFrame with columns ['Program', 'Specialty', 'College', 'Seats', 'Type']
        """
        self.college_data = college_data
        self.categories = list(SEAT_MATRIX.keys())
        self.total_seats = sum(SEAT_MATRIX.values())
        self.validate_data()
    
    def validate_data(self):
        """Validate input data"""
        required_columns = ['Program', 'Specialty', 'College', 'Seats', 'Type']
        for col in required_columns:
            if col not in self.college_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if (self.college_data['Seats'] <= 0).any():
            raise ValueError("All seat values must be positive")
    
    def hamilton_rounding(self, proportions, total_seats):
        """
        Hamilton (largest remainder) method for seat allocation
        
        Args:
            proportions: Array of proportional shares
            total_seats: Total number of seats to allocate
        
        Returns:
            Array of allocated seats
        """
        if len(proportions) == 0 or total_seats <= 0:
            return np.zeros(len(proportions), dtype=int)
        
        # Calculate initial allocation (floor)
        initial_seats = np.floor(proportions * total_seats).astype(int)
        remainder = proportions * total_seats - initial_seats
        
        # Allocate remaining seats to largest remainders
        remaining_seats = total_seats - initial_seats.sum()
        if remaining_seats > 0:
            # Get indices of largest remainders
            largest_remainder_indices = np.argsort(remainder)[-remaining_seats:]
            initial_seats[largest_remainder_indices] += 1
        
        return initial_seats
    
    def biproportional_allocation(self, row_margins, col_margins, max_iterations=1000, tolerance=1e-6):
        """
        Biproportional allocation using iterative proportional fitting
        
        Args:
            row_margins: Row marginals (colleges)
            col_margins: Column marginals (specialties)
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
        
        Returns:
            Allocated matrix
        """
        n_rows = len(row_margins)
        n_cols = len(col_margins)
        
        if n_rows == 0 or n_cols == 0:
            return np.zeros((n_rows, n_cols), dtype=int)
        
        # Initialize with uniform distribution
        matrix = np.ones((n_rows, n_cols))
        
        # Iterative proportional fitting
        for iteration in range(max_iterations):
            # Scale rows
            row_sums = matrix.sum(axis=1)
            for i in range(n_rows):
                if row_sums[i] > 0:
                    matrix[i, :] *= row_margins[i] / row_sums[i]
            
            # Scale columns
            col_sums = matrix.sum(axis=0)
            for j in range(n_cols):
                if col_sums[j] > 0:
                    matrix[:, j] *= col_margins[j] / col_sums[j]
            
            # Check convergence
            if np.allclose(matrix.sum(axis=1), row_margins, rtol=tolerance) and \
               np.allclose(matrix.sum(axis=0), col_margins, rtol=tolerance):
                break
        
        # Round to integers using Hamilton method for each row
        rounded_matrix = np.zeros_like(matrix)
        for i in range(n_rows):
            if matrix[i, :].sum() > 0:
                rounded_matrix[i, :] = self.hamilton_rounding(
                    matrix[i, :] / matrix[i, :].sum(), 
                    int(row_margins[i])
                )
        
        return rounded_matrix.astype(int)
    
    def calculate_allocations(self):
        """
        Calculate seat allocations using both methods
        
        Returns:
            Tuple of (results dictionary, colleges array, specialties array)
        """
        # Get unique colleges and specialties
        colleges = self.college_data['College'].unique()
        specialties = self.college_data['Specialty'].unique()
        
        # Create mapping for college-specialty combinations
        college_specialty_map = {}
        for _, row in self.college_data.iterrows():
            key = (row['College'], row['Specialty'])
            college_specialty_map[key] = row['Seats']
        
        # Create matrices for allocation
        n_colleges = len(colleges)
        n_specialties = len(specialties)
        
        # Total seats per college and specialty
        college_seats = np.array([
            sum(college_specialty_map.get((college, spec), 0) 
                for spec in specialties)
            for college in colleges
        ])
        
        specialty_seats = np.array([
            sum(college_specialty_map.get((college, spec), 0) 
                for college in colleges)
            for spec in specialties
        ])
        
        # Calculate category shares based on SEAT_MATRIX
        category_shares = np.array(list(SEAT_MATRIX.values())) / self.total_seats
        
        # Initialize result storage
        results = {
            'hamilton': {},
            'biproportional': {}
        }
        
        # For each category, allocate seats
        for cat_idx, category in enumerate(self.categories):
            category_total = SEAT_MATRIX[category]
            
            # Method 1: Hamilton rounding
            # Calculate proportions for each college-specialty combination
            proportions = []
            for college in colleges:
                for specialty in specialties:
                    seats = college_specialty_map.get((college, specialty), 0)
                    proportions.append(seats)
            
            proportions = np.array(proportions)
            total_seats_for_distribution = sum(proportions)
            
            if total_seats_for_distribution > 0:
                ham_allocation = self.hamilton_rounding(
                    proportions / total_seats_for_distribution, 
                    category_total
                )
            else:
                ham_allocation = np.zeros(len(proportions))
            
            # Reshape to college-specialty matrix
            ham_matrix = ham_allocation.reshape(n_colleges, n_specialties)
            
            # Method 2: Biproportional allocation
            # Create row and column margins for this category
            if n_colleges > 0 and n_specialties > 0:
                row_margins = college_seats * category_shares[cat_idx]
                col_margins = specialty_seats * category_shares[cat_idx]
                
                # Ensure margins sum to category total
                if row_margins.sum() > 0:
                    row_margins = row_margins / row_margins.sum() * category_total
                if col_margins.sum() > 0:
                    col_margins = col_margins / col_margins.sum() * category_total
                
                # Round margins to integers
                row_margins = np.round(row_margins).astype(int)
                col_margins = np.round(col_margins).astype(int)
                
                # Adjust to ensure sum equals category total
                while row_margins.sum() != category_total and row_margins.sum() > 0:
                    if row_margins.sum() < category_total:
                        row_margins[np.argmax(college_seats)] += 1
                    else:
                        row_margins[np.argmin(college_seats)] -= 1
                
                while col_margins.sum() != category_total and col_margins.sum() > 0:
                    if col_margins.sum() < category_total:
                        col_margins[np.argmax(specialty_seats)] += 1
                    else:
                        col_margins[np.argmin(specialty_seats)] -= 1
                
                # Create biproportional matrix
                bipro_matrix = self.biproportional_allocation(row_margins, col_margins)
            else:
                bipro_matrix = np.zeros((n_colleges, n_specialties), dtype=int)
            
            # Store results
            results['hamilton'][category] = ham_matrix
            results['biproportional'][category] = bipro_matrix
        
        return results, colleges, specialties

# ============================================================================
# UI FUNCTIONS
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'colleges' not in st.session_state:
        st.session_state.colleges = None
    if 'specialties' not in st.session_state:
        st.session_state.specialties = None
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    if 'history' not in st.session_state:
        st.session_state.history = []

def display_metric_cards(total_seats, n_colleges, n_specialties):
    """Display metric cards in a row"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Seats</h4>
            <h2>{total_seats}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Categories</h4>
            <h2>{len(SEAT_MATRIX)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Colleges</h4>
            <h2>{n_colleges}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Specialties</h4>
            <h2>{n_specialties}</h2>
        </div>
        """, unsafe_allow_html=True)

def display_results_tabs(results, colleges, specialties):
    """Display results in organized tabs"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Hamilton Method",
        "📈 Biproportional Method",
        "📉 Comparison",
        "📋 Detailed Breakdown",
        "📊 Visualizations"
    ])
    
    with tab1:
        st.markdown("### Hamilton Rounding Method")
        st.markdown("""
        The Hamilton method (also known as the largest remainder method) allocates seats by:
        1. Calculating each entity's fair share
        2. Giving each entity its floor allocation
        3. Distributing remaining seats to entities with the largest fractional remainders
        """)
        
        # Summary table
        ham_summary = []
        for category, matrix in results['hamilton'].items():
            total = matrix.sum()
            ham_summary.append({
                'Category': category,
                'Allocated': total,
                'Expected': SEAT_MATRIX[category],
                'Difference': total - SEAT_MATRIX[category],
                'Percentage': f"{(total/SEAT_MATRIX[category]*100):.1f}%" if SEAT_MATRIX[category] > 0 else "N/A"
            })
        
        st.dataframe(
            pd.DataFrame(ham_summary),
            use_container_width=True
        )
        
        # Detailed matrices
        with st.expander("View Detailed Allocation Matrices"):
            for category, matrix in results['hamilton'].items():
                st.markdown(f"**Category: {category}**")
                df = pd.DataFrame(
                    matrix,
                    index=colleges,
                    columns=specialties
                )
                st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("### Biproportional Method")
        st.markdown("""
        The biproportional method uses iterative proportional fitting to:
        1. Balance row (college) and column (specialty) constraints
        2. Maintain proportional representation across both dimensions
        3. Converge to a matrix that satisfies both row and column marginals
        """)
        
        bipro_summary = []
        for category, matrix in results['biproportional'].items():
            total = matrix.sum()
            bipro_summary.append({
                'Category': category,
                'Allocated': total,
                'Expected': SEAT_MATRIX[category],
                'Difference': total - SEAT_MATRIX[category],
                'Percentage': f"{(total/SEAT_MATRIX[category]*100):.1f}%" if SEAT_MATRIX[category] > 0 else "N/A"
            })
        
        st.dataframe(
            pd.DataFrame(bipro_summary),
            use_container_width=True
        )
        
        with st.expander("View Detailed Allocation Matrices"):
            for category, matrix in results['biproportional'].items():
                st.markdown(f"**Category: {category}**")
                df = pd.DataFrame(
                    matrix,
                    index=colleges,
                    columns=specialties
                )
                st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.markdown("### Method Comparison")
        
        # Comparison table
        comparison_data = []
        for category in SEAT_MATRIX.keys():
            ham_total = results['hamilton'][category].sum()
            bipro_total = results['biproportional'][category].sum()
            expected = SEAT_MATRIX[category]
            
            comparison_data.append({
                'Category': category,
                'Expected': expected,
                'Hamilton': ham_total,
                'Biproportional': bipro_total,
                'Hamilton Diff': ham_total - expected,
                'Biproportional Diff': bipro_total - expected,
                'Better Method': 'Hamilton' if abs(ham_total - expected) < abs(bipro_total - expected) else 'Biproportional'
            })
        
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True)
        
        # Comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_comp['Category'],
            y=df_comp['Expected'],
            name='Expected',
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            x=df_comp['Category'],
            y=df_comp['Hamilton'],
            name='Hamilton',
            marker_color='#2ca02c'
        ))
        fig.add_trace(go.Bar(
            x=df_comp['Category'],
            y=df_comp['Biproportional'],
            name='Biproportional',
            marker_color='#ff7f0e'
        ))
        fig.update_layout(
            title='Seat Allocation Comparison by Category',
            xaxis_title='Category',
            yaxis_title='Number of Seats',
            barmode='group',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Difference analysis
        st.markdown("#### Difference Analysis")
        diff_data = []
        for category in SEAT_MATRIX.keys():
            ham_total = results['hamilton'][category].sum()
            bipro_total = results['biproportional'][category].sum()
            expected = SEAT_MATRIX[category]
            
            diff_data.append({
                'Category': category,
                'Hamilton Deviation': ham_total - expected,
                'Biproportional Deviation': bipro_total - expected,
                'Method Difference': ham_total - bipro_total
            })
        
        df_diff = pd.DataFrame(diff_data)
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(
            x=df_diff['Category'],
            y=df_diff['Hamilton Deviation'],
            name='Hamilton Deviation',
            marker_color='#2ca02c'
        ))
        fig_diff.add_trace(go.Bar(
            x=df_diff['Category'],
            y=df_diff['Biproportional Deviation'],
            name='Biproportional Deviation',
            marker_color='#ff7f0e'
        ))
        fig_diff.update_layout(
            title='Deviation from Expected Values',
            xaxis_title='Category',
            yaxis_title='Deviation',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_diff, use_container_width=True)
    
    with tab4:
        st.markdown("### Detailed Breakdown")
        
        # College-wise summary
        st.markdown("#### College-wise Summary")
        college_summary = []
        for college in colleges:
            for method in ['hamilton', 'biproportional']:
                total = 0
                for category in SEAT_MATRIX.keys():
                    matrix = results[method][category]
                    college_idx = list(colleges).index(college)
                    total += matrix[college_idx, :].sum()
                college_summary.append({
                    'College': college,
                    'Method': method.capitalize(),
                    'Total Seats': int(total)
                })
        
        df_college = pd.DataFrame(college_summary)
        pivot_college = df_college.pivot(index='College', columns='Method', values='Total Seats')
        pivot_college['Difference'] = pivot_college['Hamilton'] - pivot_college['Biproportional']
        st.dataframe(pivot_college, use_container_width=True)
        
        # Specialty-wise summary
        st.markdown("#### Specialty-wise Summary")
        specialty_summary = []
        for specialty in specialties:
            for method in ['hamilton', 'biproportional']:
                total = 0
                for category in SEAT_MATRIX.keys():
                    matrix = results[method][category]
                    specialty_idx = list(specialties).index(specialty)
                    total += matrix[:, specialty_idx].sum()
                specialty_summary.append({
                    'Specialty': specialty,
                    'Method': method.capitalize(),
                    'Total Seats': int(total)
                })
        
        df_specialty = pd.DataFrame(specialty_summary)
        pivot_specialty = df_specialty.pivot(index='Specialty', columns='Method', values='Total Seats')
        pivot_specialty['Difference'] = pivot_specialty['Hamilton'] - pivot_specialty['Biproportional']
        st.dataframe(pivot_specialty, use_container_width=True)
        
        # Download results
        st.markdown("#### Download Results")
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # Create Excel-like download
            download_data = []
            for method in ['hamilton', 'biproportional']:
                for category in SEAT_MATRIX.keys():
                    matrix = results[method][category]
                    for i, college in enumerate(colleges):
                        for j, specialty in enumerate(specialties):
                            if matrix[i, j] > 0:
                                download_data.append({
                                    'Method': method.capitalize(),
                                    'Category': category,
                                    'College': college,
                                    'Specialty': specialty,
                                    'Seats': int(matrix[i, j])
                                })
            
            df_download = pd.DataFrame(download_data)
            csv = df_download.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"seat_allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col_download2:
            # JSON download
            json_data = {
                'timestamp': datetime.now().isoformat(),
                'seat_matrix': SEAT_MATRIX,
                'colleges': list(colleges),
                'specialties': list(specialties),
                'results': {
                    method: {
                        category: matrix.tolist() 
                        for category, matrix in results[method].items()
                    }
                    for method in ['hamilton', 'biproportional']
                }
            }
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"seat_allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with tab5:
        st.markdown("### Interactive Visualizations")
        
        # Visualization options
        viz_method = st.selectbox(
            "Select allocation method",
            ['Hamilton', 'Biproportional']
        )
        
        viz_type = st.selectbox(
            "Select visualization type",
            ['Heatmap', '3D Surface', 'Sunburst', 'Treemap', 'Bar Chart']
        )
        
        method_key = viz_method.lower()
        
        if viz_type == 'Heatmap':
            # Show heatmaps for all categories
            cols = st.columns(3)
            for idx, (category, matrix) in enumerate(results[method_key].items()):
                if idx < 3:
                    with cols[idx % 3]:
                        fig = px.imshow(
                            matrix,
                            x=specialties,
                            y=colleges,
                            title=f"{category} - {viz_method}",
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Show remaining categories if any
            if len(results[method_key]) > 3:
                st.markdown("#### Additional Categories")
                cols = st.columns(3)
                for idx, (category, matrix) in enumerate(list(results[method_key].items())[3:]):
                    with cols[idx % 3]:
                        fig = px.imshow(
                            matrix,
                            x=specialties,
                            y=colleges,
                            title=f"{category} - {viz_method}",
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == '3D Surface':
            # Create 3D surface plot
            category = st.selectbox(
                "Select category for 3D view",
                list(SEAT_MATRIX.keys())
            )
            matrix = results[method_key][category]
            
            fig = go.Figure(data=[go.Surface(
                z=matrix,
                x=list(range(len(specialties))),
                y=list(range(len(colleges))),
                colorscale='Viridis'
            )])
            fig.update_layout(
                title=f"{category} - {viz_method} Method (3D Surface)",
                scene=dict(
                    xaxis_title='Specialty Index',
                    yaxis_title='College Index',
                    zaxis_title='Seats'
                ),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == 'Sunburst':
            # Prepare data for sunburst
            sunburst_data = []
            for category, matrix in results[method_key].items():
                for i, college in enumerate(colleges):
                    for j, specialty in enumerate(specialties):
                        seats = matrix[i, j]
                        if seats > 0:
                            sunburst_data.append({
                                'Category': category,
                                'College': college,
                                'Specialty': specialty,
                                'Seats': seats
                            })
            
            df_sunburst = pd.DataFrame(sunburst_data)
            if not df_sunburst.empty:
                fig = px.sunburst(
                    df_sunburst,
                    path=['Category', 'College', 'Specialty'],
                    values='Seats',
                    title=f'Seat Distribution - {viz_method} Method',
                    color='Seats',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for sunburst chart")
        
        elif viz_type == 'Treemap':
            # Prepare data for treemap
            treemap_data = []
            for category, matrix in results[method_key].items():
                for i, college in enumerate(colleges):
                    for j, specialty in enumerate(specialties):
                        seats = matrix[i, j]
                        if seats > 0:
                            treemap_data.append({
                                'Category': category,
                                'College': college,
                                'Specialty': specialty,
                                'Seats': seats
                            })
            
            df_treemap = pd.DataFrame(treemap_data)
            if not df_treemap.empty:
                fig = px.treemap(
                    df_treemap,
                    path=['Category', 'College', 'Specialty'],
                    values='Seats',
                    title=f'Seat Distribution - {viz_method} Method',
                    color='Seats',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for treemap")
        
        elif viz_type == 'Bar Chart':
            # Bar chart visualization
            category = st.selectbox(
                "Select category for bar chart",
                list(SEAT_MATRIX.keys())
            )
            matrix = results[method_key][category]
            
            # Prepare data
            bar_data = []
            for i, college in enumerate(colleges):
                for j, specialty in enumerate(specialties):
                    seats = matrix[i, j]
                    if seats > 0:
                        bar_data.append({
                            'College': college,
                            'Specialty': specialty,
                            'Seats': seats
                        })
            
            df_bar = pd.DataFrame(bar_data)
            if not df_bar.empty:
                fig = px.bar(
                    df_bar,
                    x='College',
                    y='Seats',
                    color='Specialty',
                    title=f'{category} - {viz_method} Method',
                    barmode='group',
                    text='Seats'
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎓 Seat Allocation System</div>', unsafe_allow_html=True)
    st.markdown("### Hamilton Rounding vs Biproportional Method")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Configuration")
        
        # Display category seats
        st.subheader("Category Seats")
        df_categories = pd.DataFrame(
            list(SEAT_MATRIX.items()),
            columns=['Category', 'Seats']
        )
        st.dataframe(df_categories, use_container_width=True)
        
        st.subheader("College Data Input")
        
        # Input options
        input_option = st.radio(
            "Select input method:",
            ['Use Sample Data', 'Upload CSV', 'Manual Entry'],
            help="Choose how to provide college data"
        )
        
        college_data = None
        
        if input_option == 'Use Sample Data':
            # Sample data with more colleges
            sample_data = pd.DataFrame({
                'Program': ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E'],
                'Specialty': ['DS', 'DS', 'DS', 'AI', 'AI', 'CS', 'CS', 'CS', 'ML', 'ML', 'DS', 'AI'],
                'College': ['CDI', 'CDP', 'CDT', 'CDI', 'CDP', 'CDT', 'CDI', 'CDP', 'CDT', 'CDI', 'CDP', 'CDT'],
                'Seats': [1, 1, 4, 2, 3, 1, 2, 1, 2, 1, 1, 2],
                'Type': ['G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G']
            })
            college_data = sample_data
            st.dataframe(sample_data, use_container_width=True)
            
        elif input_option == 'Upload CSV':
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="CSV must have columns: Program, Specialty, College, Seats, Type"
            )
            if uploaded_file is not None:
                college_data = pd.read_csv(uploaded_file)
                st.dataframe(college_data, use_container_width=True)
            else:
                st.info("Please upload a CSV file")
                
        elif input_option == 'Manual Entry':
            st.info("Enter data manually below")
            n_rows = st.number_input("Number of entries", min_value=1, max_value=20, value=3)
            
            # Create editable dataframe
            manual_data = []
            for i in range(n_rows):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    program = st.text_input(f"Program {i+1}", value="E", key=f"prog_{i}")
                with col2:
                    specialty = st.text_input(f"Specialty {i+1}", value="DS", key=f"spec_{i}")
                with col3:
                    college = st.text_input(f"College {i+1}", value=f"CD{i+1}", key=f"col_{i}")
                with col4:
                    seats = st.number_input(f"Seats {i+1}", min_value=1, value=1, key=f"seat_{i}")
                with col5:
                    type_val = st.text_input(f"Type {i+1}", value="G", key=f"type_{i}")
                manual_data.append({
                    'Program': program,
                    'Specialty': specialty,
                    'College': college,
                    'Seats': seats,
                    'Type': type_val
                })
            
            if manual_data:
                college_data = pd.DataFrame(manual_data)
        
        # Calculate button
        if st.button("🚀 Calculate Allocations", type="primary", use_container_width=True):
            if college_data is not None and not college_data.empty:
                try:
                    with st.spinner("Calculating allocations..."):
                        allocator = SeatAllocator(college_data)
                        results, colleges, specialties = allocator.calculate_allocations()
                        
                        st.session_state.results = results
                        st.session_state.colleges = colleges
                        st.session_state.specialties = specialties
                        st.session_state.calculated = True
                        
                        st.success("✅ Allocations calculated successfully!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Error in calculation: {str(e)}")
                    st.exception(e)
            else:
                st.warning("Please provide valid college data")
    
    # Main content area
    if st.session_state.calculated and st.session_state.results is not None:
        # Display metrics
        display_metric_cards(
            sum(SEAT_MATRIX.values()),
            len(st.session_state.colleges),
            len(st.session_state.specialties)
        )
        
        # Display results
        display_results_tabs(
            st.session_state.results,
            st.session_state.colleges,
            st.session_state.specialties
        )
        
    else:
        # Welcome message when no calculation done
        st.markdown("""
        ### Welcome to the Seat Allocation System! 👋
        
        This application helps you allocate seats across colleges and specialties using two different methods:
        
        #### 🎯 Hamilton Rounding Method
        - Allocates seats based on proportional representation
        - Uses the largest remainder method for fair distribution
        - Simple and transparent allocation process
        
        #### 📊 Biproportional Method
        - Balances allocation across both colleges and specialties
        - Uses iterative proportional fitting
        - Ensures both row and column constraints are satisfied
        
        #### How to get started:
        1. Configure your data in the sidebar
        2. Click "Calculate Allocations"
        3. Explore the results in the tabs above
        
        #### Features:
        - ✅ Multiple input methods (sample, CSV, manual)
        - ✅ Visual comparisons between methods
        - ✅ Export results (CSV, JSON)
        - ✅ Interactive visualizations
        - ✅ Detailed breakdown by college and specialty
        
        **Get started by configuring your data in the sidebar!**
        """)
        
        # Sample visualization preview
        st.markdown("### 🎨 Method Comparison Preview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Hamilton Method** ✅
            - Simple and transparent
            - Based on largest remainders
            - Easy to understand
            - Good for simple allocations
            """)
        
        with col2:
            st.markdown("""
            **Biproportional Method** ✅
            - Balances multiple constraints
            - Iterative fitting algorithm
            - More complex but precise
            - Handles complex scenarios
            """)
        
        with col3:
            st.markdown("""
            **Comparison Features** 📊
            - Side-by-side comparison
            - Difference analysis
            - Visual representations
            - Export capabilities
            """)
        
        # Example visualization
        st.markdown("### 📈 Example Visualization")
        example_data = {
            'Category': ['SM', 'EW', 'EZ', 'MU', 'SC', 'BH', 'LA', 'DV', 'VK', 'ST', 'KN', 'BX', 'KU'],
            'Seats': [50, 10, 9, 8, 8, 3, 3, 2, 2, 2, 1, 1, 1]
        }
        df_example = pd.DataFrame(example_data)
        fig = px.pie(df_example, values='Seats', names='Category', title='Seat Distribution by Category')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
