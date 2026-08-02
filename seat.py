import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
import plotly.express as px
import plotly.graph_objects as go

# Define the seat matrix based on categories
SEAT_MATRIX = {
    'SM': 50, 'EW': 10, 'EZ': 9, 'MU': 8, 'SC': 8, 
    'BH': 3, 'LA': 3, 'DV': 2, 'VK': 2, 'ST': 2, 
    'KN': 1, 'BX': 1, 'KU': 1
}

class SeatAllocator:
    def __init__(self, college_data):
        """
        Initialize with college data
        college_data: DataFrame with columns ['Program', 'Specialty', 'College', 'Seats', 'Type']
        """
        self.college_data = college_data
        self.categories = list(SEAT_MATRIX.keys())
        self.total_seats = sum(SEAT_MATRIX.values())
        
    def hamilton_rounding(self, proportions, total_seats):
        """
        Hamilton (largest remainder) method for seat allocation
        """
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
    
    def biproportional_allocation(self, row_margins, col_margins, max_iterations=1000):
        """
        Biproportional allocation using iterative proportional fitting
        """
        # Initialize with uniform distribution
        n_rows = len(row_margins)
        n_cols = len(col_margins)
        
        # Initial matrix with equal proportions
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
            if np.allclose(matrix.sum(axis=1), row_margins, rtol=1e-6) and \
               np.allclose(matrix.sum(axis=0), col_margins, rtol=1e-6):
                break
        
        # Round to integers using Hamilton method for each row
        rounded_matrix = np.zeros_like(matrix)
        for i in range(n_rows):
            rounded_matrix[i, :] = self.hamilton_rounding(
                matrix[i, :] / matrix[i, :].sum(), 
                int(row_margins[i])
            )
        
        return rounded_matrix.astype(int)
    
    def calculate_allocations(self):
        """
        Calculate seat allocations using both methods
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
        n_categories = len(self.categories)
        
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
            
            # Method 1: Hamilton rounding - allocate based on college specialty seats
            # Calculate proportions for each college-specialty combination
            proportions = []
            for college in colleges:
                for specialty in specialties:
                    seats = college_specialty_map.get((college, specialty), 0)
                    proportions.append(seats)
            
            proportions = np.array(proportions)
            total_seats_for_distribution = sum(proportions)
            
            if total_seats_for_distribution > 0:
                # Hamilton allocation across college-specialty combinations
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
            row_margins = college_seats * category_shares[cat_idx]
            col_margins = specialty_seats * category_shares[cat_idx]
            
            # Ensure margins sum to category total
            row_margins = row_margins / row_margins.sum() * category_total
            col_margins = col_margins / col_margins.sum() * category_total
            
            # Round margins to integers
            row_margins = np.round(row_margins).astype(int)
            col_margins = np.round(col_margins).astype(int)
            
            # Adjust to ensure sum equals category total
            while row_margins.sum() != category_total:
                if row_margins.sum() < category_total:
                    row_margins[np.argmax(college_seats)] += 1
                else:
                    row_margins[np.argmin(college_seats)] -= 1
            
            # Create initial biproportional matrix
            if n_colleges > 0 and n_specialties > 0:
                bipro_matrix = self.biproportional_allocation(row_margins, col_margins)
            else:
                bipro_matrix = np.zeros((n_colleges, n_specialties))
            
            # Store results
            results['hamilton'][category] = ham_matrix
            results['biproportional'][category] = bipro_matrix
        
        return results, colleges, specialties

def create_visualization(results, colleges, specialties):
    """
    Create visualizations for the allocation results
    """
    st.subheader("📊 Allocation Visualization")
    
    # Prepare data for visualization
    viz_data = []
    for method in ['hamilton', 'biproportional']:
        for category in SEAT_MATRIX.keys():
            matrix = results[method][category]
            for i, college in enumerate(colleges):
                for j, specialty in enumerate(specialties):
                    if matrix[i, j] > 0:
                        viz_data.append({
                            'Method': method.capitalize(),
                            'Category': category,
                            'College': college,
                            'Specialty': specialty,
                            'Seats': matrix[i, j]
                        })
    
    df_viz = pd.DataFrame(viz_data)
    
    if not df_viz.empty:
        # Create bar chart for comparison
        fig1 = px.bar(
            df_viz, 
            x='Category', 
            y='Seats', 
            color='College',
            facet_col='Method',
            title='Seat Allocation by Category and Method',
            barmode='group'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Heatmap for one method
        method_choice = st.selectbox(
            "Select method for heatmap visualization",
            ['hamilton', 'biproportional']
        )
        
        if method_choice in results:
            # Create heatmap for each category
            cols = st.columns(3)
            for idx, (category, matrix) in enumerate(results[method_choice].items()):
                if idx < 3:  # Show first 3 categories
                    with cols[idx % 3]:
                        fig2 = px.imshow(
                            matrix,
                            x=specialties,
                            y=colleges,
                            title=f"{category} - {method_choice.capitalize()}",
                            text_auto=True,
                            aspect="auto"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
    
    return df_viz

def main():
    st.set_page_config(
        page_title="Seat Allocation System",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Seat Allocation System")
    st.markdown("### Hamilton Rounding vs Biproportional Method")
    
    # Sidebar for input
    with st.sidebar:
        st.header("📋 Input Configuration")
        
        st.subheader("Category Seats")
        st.dataframe(
            pd.DataFrame(list(SEAT_MATRIX.items()), 
                        columns=['Category', 'Seats']),
            use_container_width=True
        )
        
        st.subheader("College Data Input")
        
        # Default input data
        default_data = {
            'Program': ['E', 'E', 'E'],
            'Specialty': ['DS', 'DS', 'DS'],
            'College': ['CDI', 'CDP', 'CDT'],
            'Seats': [1, 1, 4],
            'Type': ['G', 'G', 'G']
        }
        
        # Allow user to upload CSV or use default
        upload_option = st.radio(
            "Choose input method:",
            ['Use Default Data', 'Upload CSV']
        )
        
        if upload_option == 'Upload CSV':
            uploaded_file = st.file_uploader(
                "Upload CSV file", 
                type=['csv'],
                help="CSV should have columns: Program, Specialty, College, Seats, Type"
            )
            if uploaded_file is not None:
                college_data = pd.read_csv(uploaded_file)
            else:
                st.warning("Please upload a CSV file or switch to default data")
                return
        else:
            college_data = pd.DataFrame(default_data)
        
        st.dataframe(college_data, use_container_width=True)
        
        if st.button("🚀 Calculate Allocations", type="primary"):
            st.session_state['calculate'] = True
    
    # Main content area
    if 'calculate' in st.session_state and st.session_state['calculate']:
        try:
            # Initialize allocator
            allocator = SeatAllocator(college_data)
            
            # Calculate allocations
            with st.spinner("Calculating allocations..."):
                results, colleges, specialties = allocator.calculate_allocations()
            
            # Display results
            st.success("✅ Allocations calculated successfully!")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Seats Available", sum(SEAT_MATRIX.values()))
            with col2:
                st.metric("Total Colleges", len(colleges))
            with col3:
                st.metric("Total Specialties", len(specialties))
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Hamilton Method", 
                "📈 Biproportional Method",
                "📉 Comparison",
                "📋 Detailed Breakdown"
            ])
            
            with tab1:
                st.subheader("Hamilton Method Results")
                # Create summary DataFrame
                ham_summary = []
                for category, matrix in results['hamilton'].items():
                    total = matrix.sum()
                    ham_summary.append({
                        'Category': category,
                        'Total Allocated': total,
                        'Expected': SEAT_MATRIX[category],
                        'Difference': total - SEAT_MATRIX[category]
                    })
                st.dataframe(pd.DataFrame(ham_summary))
                
                # Show detailed matrix
                for category, matrix in results['hamilton'].items():
                    st.write(f"**{category}**")
                    df = pd.DataFrame(
                        matrix,
                        index=colleges,
                        columns=specialties
                    )
                    st.dataframe(df)
            
            with tab2:
                st.subheader("Biproportional Method Results")
                bipro_summary = []
                for category, matrix in results['biproportional'].items():
                    total = matrix.sum()
                    bipro_summary.append({
                        'Category': category,
                        'Total Allocated': total,
                        'Expected': SEAT_MATRIX[category],
                        'Difference': total - SEAT_MATRIX[category]
                    })
                st.dataframe(pd.DataFrame(bipro_summary))
                
                for category, matrix in results['biproportional'].items():
                    st.write(f"**{category}**")
                    df = pd.DataFrame(
                        matrix,
                        index=colleges,
                        columns=specialties
                    )
                    st.dataframe(df)
            
            with tab3:
                # Comparison visualization
                st.subheader("Method Comparison")
                
                # Create comparison DataFrame
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
                        'Hamilton Difference': ham_total - expected,
                        'Biproportional Difference': bipro_total - expected
                    })
                
                df_comp = pd.DataFrame(comparison_data)
                st.dataframe(df_comp)
                
                # Create visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_comp['Category'],
                    y=df_comp['Expected'],
                    name='Expected',
                    marker_color='blue'
                ))
                fig.add_trace(go.Bar(
                    x=df_comp['Category'],
                    y=df_comp['Hamilton'],
                    name='Hamilton',
                    marker_color='green'
                ))
                fig.add_trace(go.Bar(
                    x=df_comp['Category'],
                    y=df_comp['Biproportional'],
                    name='Biproportional',
                    marker_color='orange'
                ))
                fig.update_layout(
                    title='Seat Allocation Comparison',
                    xaxis_title='Category',
                    yaxis_title='Number of Seats',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Detailed Breakdown")
                
                # Show college-wise summary
                st.write("### College-wise Summary")
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
                            'Total Seats': total
                        })
                st.dataframe(pd.DataFrame(college_summary))
                
                # Show specialty-wise summary
                st.write("### Specialty-wise Summary")
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
                            'Total Seats': total
                        })
                st.dataframe(pd.DataFrame(specialty_summary))
            
            # Additional visualizations
            create_visualization(results, colleges, specialties)
            
        except Exception as e:
            st.error(f"Error in calculation: {str(e)}")
            st.exception(e)
    
    # Instructions
    with st.expander("ℹ️ How to use this application"):
        st.markdown("""
        ### Seat Allocation System
        
        This application implements two methods for seat allocation:
        
        1. **Hamilton Rounding Method**: 
           - Allocates seats based on proportions using the largest remainder method
           - Ensures each category gets at least its floor allocation
        
        2. **Biproportional Method**:
           - Uses iterative proportional fitting
           - Balances allocation across both row (college) and column (specialty) constraints
        
        ### Input Format
        - Upload a CSV file with columns: Program, Specialty, College, Seats, Type
        - Or use the default data provided
        
        ### Output
        - Detailed allocation tables for each method
        - Visual comparisons
        - College-wise and specialty-wise summaries
        """)

if __name__ == "__main__":
    main()import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
import plotly.express as px
import plotly.graph_objects as go

# Define the seat matrix based on categories
SEAT_MATRIX = {
    'SM': 50, 'EW': 10, 'EZ': 9, 'MU': 8, 'SC': 8, 
    'BH': 3, 'LA': 3, 'DV': 2, 'VK': 2, 'ST': 2, 
    'KN': 1, 'BX': 1, 'KU': 1
}

class SeatAllocator:
    def __init__(self, college_data):
        """
        Initialize with college data
        college_data: DataFrame with columns ['Program', 'Specialty', 'College', 'Seats', 'Type']
        """
        self.college_data = college_data
        self.categories = list(SEAT_MATRIX.keys())
        self.total_seats = sum(SEAT_MATRIX.values())
        
    def hamilton_rounding(self, proportions, total_seats):
        """
        Hamilton (largest remainder) method for seat allocation
        """
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
    
    def biproportional_allocation(self, row_margins, col_margins, max_iterations=1000):
        """
        Biproportional allocation using iterative proportional fitting
        """
        # Initialize with uniform distribution
        n_rows = len(row_margins)
        n_cols = len(col_margins)
        
        # Initial matrix with equal proportions
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
            if np.allclose(matrix.sum(axis=1), row_margins, rtol=1e-6) and \
               np.allclose(matrix.sum(axis=0), col_margins, rtol=1e-6):
                break
        
        # Round to integers using Hamilton method for each row
        rounded_matrix = np.zeros_like(matrix)
        for i in range(n_rows):
            rounded_matrix[i, :] = self.hamilton_rounding(
                matrix[i, :] / matrix[i, :].sum(), 
                int(row_margins[i])
            )
        
        return rounded_matrix.astype(int)
    
    def calculate_allocations(self):
        """
        Calculate seat allocations using both methods
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
        n_categories = len(self.categories)
        
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
            
            # Method 1: Hamilton rounding - allocate based on college specialty seats
            # Calculate proportions for each college-specialty combination
            proportions = []
            for college in colleges:
                for specialty in specialties:
                    seats = college_specialty_map.get((college, specialty), 0)
                    proportions.append(seats)
            
            proportions = np.array(proportions)
            total_seats_for_distribution = sum(proportions)
            
            if total_seats_for_distribution > 0:
                # Hamilton allocation across college-specialty combinations
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
            row_margins = college_seats * category_shares[cat_idx]
            col_margins = specialty_seats * category_shares[cat_idx]
            
            # Ensure margins sum to category total
            row_margins = row_margins / row_margins.sum() * category_total
            col_margins = col_margins / col_margins.sum() * category_total
            
            # Round margins to integers
            row_margins = np.round(row_margins).astype(int)
            col_margins = np.round(col_margins).astype(int)
            
            # Adjust to ensure sum equals category total
            while row_margins.sum() != category_total:
                if row_margins.sum() < category_total:
                    row_margins[np.argmax(college_seats)] += 1
                else:
                    row_margins[np.argmin(college_seats)] -= 1
            
            # Create initial biproportional matrix
            if n_colleges > 0 and n_specialties > 0:
                bipro_matrix = self.biproportional_allocation(row_margins, col_margins)
            else:
                bipro_matrix = np.zeros((n_colleges, n_specialties))
            
            # Store results
            results['hamilton'][category] = ham_matrix
            results['biproportional'][category] = bipro_matrix
        
        return results, colleges, specialties

def create_visualization(results, colleges, specialties):
    """
    Create visualizations for the allocation results
    """
    st.subheader("📊 Allocation Visualization")
    
    # Prepare data for visualization
    viz_data = []
    for method in ['hamilton', 'biproportional']:
        for category in SEAT_MATRIX.keys():
            matrix = results[method][category]
            for i, college in enumerate(colleges):
                for j, specialty in enumerate(specialties):
                    if matrix[i, j] > 0:
                        viz_data.append({
                            'Method': method.capitalize(),
                            'Category': category,
                            'College': college,
                            'Specialty': specialty,
                            'Seats': matrix[i, j]
                        })
    
    df_viz = pd.DataFrame(viz_data)
    
    if not df_viz.empty:
        # Create bar chart for comparison
        fig1 = px.bar(
            df_viz, 
            x='Category', 
            y='Seats', 
            color='College',
            facet_col='Method',
            title='Seat Allocation by Category and Method',
            barmode='group'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Heatmap for one method
        method_choice = st.selectbox(
            "Select method for heatmap visualization",
            ['hamilton', 'biproportional']
        )
        
        if method_choice in results:
            # Create heatmap for each category
            cols = st.columns(3)
            for idx, (category, matrix) in enumerate(results[method_choice].items()):
                if idx < 3:  # Show first 3 categories
                    with cols[idx % 3]:
                        fig2 = px.imshow(
                            matrix,
                            x=specialties,
                            y=colleges,
                            title=f"{category} - {method_choice.capitalize()}",
                            text_auto=True,
                            aspect="auto"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
    
    return df_viz

def main():
    st.set_page_config(
        page_title="Seat Allocation System",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Seat Allocation System")
    st.markdown("### Hamilton Rounding vs Biproportional Method")
    
    # Sidebar for input
    with st.sidebar:
        st.header("📋 Input Configuration")
        
        st.subheader("Category Seats")
        st.dataframe(
            pd.DataFrame(list(SEAT_MATRIX.items()), 
                        columns=['Category', 'Seats']),
            use_container_width=True
        )
        
        st.subheader("College Data Input")
        
        # Default input data
        default_data = {
            'Program': ['E', 'E', 'E'],
            'Specialty': ['DS', 'DS', 'DS'],
            'College': ['CDI', 'CDP', 'CDT'],
            'Seats': [1, 1, 4],
            'Type': ['G', 'G', 'G']
        }
        
        # Allow user to upload CSV or use default
        upload_option = st.radio(
            "Choose input method:",
            ['Use Default Data', 'Upload CSV']
        )
        
        if upload_option == 'Upload CSV':
            uploaded_file = st.file_uploader(
                "Upload CSV file", 
                type=['csv'],
                help="CSV should have columns: Program, Specialty, College, Seats, Type"
            )
            if uploaded_file is not None:
                college_data = pd.read_csv(uploaded_file)
            else:
                st.warning("Please upload a CSV file or switch to default data")
                return
        else:
            college_data = pd.DataFrame(default_data)
        
        st.dataframe(college_data, use_container_width=True)
        
        if st.button("🚀 Calculate Allocations", type="primary"):
            st.session_state['calculate'] = True
    
    # Main content area
    if 'calculate' in st.session_state and st.session_state['calculate']:
        try:
            # Initialize allocator
            allocator = SeatAllocator(college_data)
            
            # Calculate allocations
            with st.spinner("Calculating allocations..."):
                results, colleges, specialties = allocator.calculate_allocations()
            
            # Display results
            st.success("✅ Allocations calculated successfully!")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Seats Available", sum(SEAT_MATRIX.values()))
            with col2:
                st.metric("Total Colleges", len(colleges))
            with col3:
                st.metric("Total Specialties", len(specialties))
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Hamilton Method", 
                "📈 Biproportional Method",
                "📉 Comparison",
                "📋 Detailed Breakdown"
            ])
            
            with tab1:
                st.subheader("Hamilton Method Results")
                # Create summary DataFrame
                ham_summary = []
                for category, matrix in results['hamilton'].items():
                    total = matrix.sum()
                    ham_summary.append({
                        'Category': category,
                        'Total Allocated': total,
                        'Expected': SEAT_MATRIX[category],
                        'Difference': total - SEAT_MATRIX[category]
                    })
                st.dataframe(pd.DataFrame(ham_summary))
                
                # Show detailed matrix
                for category, matrix in results['hamilton'].items():
                    st.write(f"**{category}**")
                    df = pd.DataFrame(
                        matrix,
                        index=colleges,
                        columns=specialties
                    )
                    st.dataframe(df)
            
            with tab2:
                st.subheader("Biproportional Method Results")
                bipro_summary = []
                for category, matrix in results['biproportional'].items():
                    total = matrix.sum()
                    bipro_summary.append({
                        'Category': category,
                        'Total Allocated': total,
                        'Expected': SEAT_MATRIX[category],
                        'Difference': total - SEAT_MATRIX[category]
                    })
                st.dataframe(pd.DataFrame(bipro_summary))
                
                for category, matrix in results['biproportional'].items():
                    st.write(f"**{category}**")
                    df = pd.DataFrame(
                        matrix,
                        index=colleges,
                        columns=specialties
                    )
                    st.dataframe(df)
            
            with tab3:
                # Comparison visualization
                st.subheader("Method Comparison")
                
                # Create comparison DataFrame
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
                        'Hamilton Difference': ham_total - expected,
                        'Biproportional Difference': bipro_total - expected
                    })
                
                df_comp = pd.DataFrame(comparison_data)
                st.dataframe(df_comp)
                
                # Create visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_comp['Category'],
                    y=df_comp['Expected'],
                    name='Expected',
                    marker_color='blue'
                ))
                fig.add_trace(go.Bar(
                    x=df_comp['Category'],
                    y=df_comp['Hamilton'],
                    name='Hamilton',
                    marker_color='green'
                ))
                fig.add_trace(go.Bar(
                    x=df_comp['Category'],
                    y=df_comp['Biproportional'],
                    name='Biproportional',
                    marker_color='orange'
                ))
                fig.update_layout(
                    title='Seat Allocation Comparison',
                    xaxis_title='Category',
                    yaxis_title='Number of Seats',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Detailed Breakdown")
                
                # Show college-wise summary
                st.write("### College-wise Summary")
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
                            'Total Seats': total
                        })
                st.dataframe(pd.DataFrame(college_summary))
                
                # Show specialty-wise summary
                st.write("### Specialty-wise Summary")
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
                            'Total Seats': total
                        })
                st.dataframe(pd.DataFrame(specialty_summary))
            
            # Additional visualizations
            create_visualization(results, colleges, specialties)
            
        except Exception as e:
            st.error(f"Error in calculation: {str(e)}")
            st.exception(e)
    
    # Instructions
    with st.expander("ℹ️ How to use this application"):
        st.markdown("""
        ### Seat Allocation System
        
        This application implements two methods for seat allocation:
        
        1. **Hamilton Rounding Method**: 
           - Allocates seats based on proportions using the largest remainder method
           - Ensures each category gets at least its floor allocation
        
        2. **Biproportional Method**:
           - Uses iterative proportional fitting
           - Balances allocation across both row (college) and column (specialty) constraints
        
        ### Input Format
        - Upload a CSV file with columns: Program, Specialty, College, Seats, Type
        - Or use the default data provided
        
        ### Output
        - Detailed allocation tables for each method
        - Visual comparisons
        - College-wise and specialty-wise summaries
        """)

if __name__ == "__main__":
    main()
