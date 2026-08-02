import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# Define the seat matrix based on categories
SEAT_MATRIX = {
    'SM': 50, 'EW': 10, 'EZ': 9, 'MU': 8, 'SC': 8, 
    'BH': 3, 'LA': 3, 'DV': 2, 'VK': 2, 'ST': 2, 
    'KN': 1, 'BX': 1, 'KU': 1
}

class SeatAllocator:
    """
    Seat allocation using Hamilton rounding and biproportional methods
    """
    
    def __init__(self, college_data: pd.DataFrame):
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
    
    def hamilton_rounding(self, proportions: np.ndarray, total_seats: int) -> np.ndarray:
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
    
    def biproportional_allocation(self, row_margins: np.ndarray, col_margins: np.ndarray, 
                                  max_iterations: int = 1000, tolerance: float = 1e-6) -> np.ndarray:
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
    
    def calculate_allocations(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
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

def validate_allocations(results: Dict, colleges: np.ndarray, specialties: np.ndarray) -> Dict:
    """
    Validate the allocation results
    
    Args:
        results: Allocation results from calculate_allocations
        colleges: Array of college names
        specialties: Array of specialty names
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'hamilton': {'valid': True, 'errors': []},
        'biproportional': {'valid': True, 'errors': []}
    }
    
    for method in ['hamilton', 'biproportional']:
        total_allocated = 0
        for category, matrix in results[method].items():
            # Check if sum equals expected
            expected = SEAT_MATRIX[category]
            actual = matrix.sum()
            if actual != expected:
                validation[method]['valid'] = False
                validation[method]['errors'].append(
                    f"{category}: Expected {expected}, got {actual}"
                )
            total_allocated += actual
        
        # Check total
        if total_allocated != sum(SEAT_MATRIX.values()):
            validation[method]['valid'] = False
            validation[method]['errors'].append(
                f"Total: Expected {sum(SEAT_MATRIX.values())}, got {total_allocated}"
            )
    
    return validation
