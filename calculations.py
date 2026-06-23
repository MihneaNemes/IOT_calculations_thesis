"""Body fat calculations and categorization."""

class BodyFatCalculator:
    """Handles BIA and body fat calculations."""
    
    @staticmethod
    def calculate_bia_body_fat(height_cm, weight_kg, age, resistance_ohms, sex):
        if resistance_ohms <= 0:
            resistance_ohms = 500.0

        impedance_index = (height_cm ** 2) / resistance_ohms
        
        if sex.upper().startswith("M"):
            ffm = (0.396 * impedance_index) + (0.143 * weight_kg) + 8.4
        else:
            ffm = (0.340 * impedance_index) + (0.153 * weight_kg) + 4.5

        ffm = min(ffm, weight_kg * 0.95)
        
        fat_mass = weight_kg - ffm
        bodyfat_percent = (fat_mass / weight_kg) * 100
        
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