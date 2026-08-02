import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.allocator import SeatAllocator, SEAT_MATRIX
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Seat Allocation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

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

def save_to_history(data):
    """Save calculation results to history"""
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data': data
    }
    st.session_state.history.append(entry)

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
            use_container_width=True,
            column_config={
                "Category": "Category",
                "Allocated": st.column_config.NumberColumn("Allocated", format="%d"),
                "Expected": st.column_config.NumberColumn("Expected", format="%d"),
                "Difference": st.column_config.NumberColumn("Difference", format="%d"),
                "Percentage": "Allocation %"
            }
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
            use_container_width=True,
            column_config={
                "Category": "Category",
                "Allocated": st.column_config.NumberColumn("Allocated", format="%d"),
                "Expected": st.column_config.NumberColumn("Expected", format="%d"),
                "Difference": st.column_config.NumberColumn("Difference", format="%d"),
                "Percentage": "Allocation %"
            }
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
            ['Heatmap', '3D Surface', 'Sunburst', 'Treemap']
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
                    title=f'Seat Distribution - {viz_method} Method'
                )
                st.plotly_chart(fig, use_container_width=True)
        
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
                'Program': ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E'],
                'Specialty': ['DS', 'DS', 'DS', 'AI', 'AI', 'CS', 'CS', 'CS'],
                'College': ['CDI', 'CDP', 'CDT', 'CDI', 'CDP', 'CDT', 'CDI', 'CDP'],
                'Seats': [1, 1, 4, 2, 3, 1, 2, 1],
                'Type': ['G', 'G', 'G', 'G', 'G', 'G', 'G', 'G']
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
            st.info("Manual entry option - enter data below")
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
                        
                        # Save to history
                        save_to_history({
                            'college_data': college_data.to_dict(),
                            'results': results,
                            'colleges': list(colleges),
                            'specialties': list(specialties)
                        })
                        
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
        
        # History section
        with st.expander("📜 Calculation History"):
            if st.session_state.history:
                for i, entry in enumerate(reversed(st.session_state.history[-5:])):
                    st.markdown(f"**{entry['timestamp']}**")
                    st.json(entry['data']['college_data'])
            else:
                st.info("No calculation history yet")
    
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
        st.markdown("### 🎨 Preview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Hamilton Method**
            - Simple and transparent
            - Based on largest remainders
            - Easy to understand
            """)
        
        with col2:
            st.markdown("""
            **Biproportional Method**
            - Balances multiple constraints
            - Iterative fitting algorithm
            - More complex but precise
            """)
        
        with col3:
            st.markdown("""
            **Comparison**
            - Visual comparisons
            - Difference analysis
            - Method selection guidance
            """)

if __name__ == "__main__":
    main()
