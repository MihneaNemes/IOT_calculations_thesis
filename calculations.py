"""Body fat calculations and categorization."""

class BodyFatCalculator:
    """Handles BIA and body fat calculations."""
    
    @staticmethod
    def calculate_bia_body_fat(height_cm, weight_kg, age, resistance_ohms, sex):
        # Safety check to prevent dividing by zero if the sensor misreads
        if resistance_ohms <= 0:
            resistance_ohms = 500.0  # Default average human baseline

        # The core metric of all medical BIA machines: The Impedance Index
        impedance_index = (height_cm ** 2) / resistance_ohms
        
        # Calculate Fat-Free Mass (FFM) first, based on muscle conductivity
        if sex.upper().startswith("M"):
            # Standard Male Equation
            ffm = (0.396 * impedance_index) + (0.143 * weight_kg) + 8.4
        else:
            # Standard Female Equation
            ffm = (0.340 * impedance_index) + (0.153 * weight_kg) + 4.5

        # Safety boundary (you can't have more FFM than total weight!)
        ffm = min(ffm, weight_kg * 0.95)
        
        # Calculate Fat Mass and Percentage based on the FFM we just found
        fat_mass = weight_kg - ffm
        bodyfat_percent = (fat_mass / weight_kg) * 100
        
        # Cap the lowest possible body fat at an essential 3%
        final_bf_percent = max(3.0, bodyfat_percent)
        
        return ffm, fat_mass, final_bf_percent
    
    @staticmethod
    def get_category(body_fat_percent, sex):
        if sex.upper().startswith("M"):
            if body_fat_percent < 6: return "Essential Fat"
            elif body_fat_percent <= 13: return "Athletic"
            elif body_fat_percent <= 17: return "Fitness"
            elif body_fat_percent <= 24: return "Average"
            else: return "Obese"
        else:
            if body_fat_percent < 14: return "Essential Fat"
            elif body_fat_percent <= 20: return "Athletic"
            elif body_fat_percent <= 24: return "Fitness"
            elif body_fat_percent <= 31: return "Average"
            else: return "Obese"
            
    @staticmethod
    def hybrid_prediction(ai_body_fat, bia_body_fat):
        return (ai_body_fat + bia_body_fat) / 2