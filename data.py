import pandas as pd

def get_hotels():
    return [
        {
            "name": "Muong Thanh Luxury Nha Trang",
            "stars": 5, "match_score": 92.6,
            "address": "60 Trần Phú, Nha Trang", "reviews": 1269,
            "description": "Khách sạn cao cấp ven biển, hồ bơi ngoài trời, spa & gym, tầm nhìn toàn cảnh. Phù hợp gia đình và công tác...",
            "rating": 8.8, "location_score": 9.2, "cleanliness_score": 9.0, "service_score": 8.8,
            "price": "2.000.000 ₫", "tags": ["🏖️ Gần biển","💆‍♀️ Spa","💪 Gym","👨‍👩‍👧‍👦 Gia đình"]
        },
        {
            "name": "Aaron Hotel",
            "stars": 3, "match_score": 88.4,
            "address": "10 Trần Quang Khải, Nha Trang", "reviews": 300,
            "description": "Khách sạn trung tâm, phòng sạch sẽ, gần điểm tham quan, phù hợp cặp đôi và công tác...",
            "rating": 8.5, "location_score": 8.9, "cleanliness_score": 9.1, "service_score": 8.4,
            "price": "1.200.000 ₫", "tags": ["📍 Vị trí đẹp","✨ Sạch sẽ","💼 Business"]
        },
        {
            "name": "Panorama Star Beach",
            "stars": 5, "match_score": 87.2,
            "address": "Nguyễn Thiện Thuật, Nha Trang", "reviews": 540,
            "description": "Khu nghỉ dưỡng sát biển với hồ bơi vô cực, sky bar, phòng rộng và view biển...",
            "rating": 8.7, "location_score": 9.0, "cleanliness_score": 8.7, "service_score": 8.6,
            "price": "2.100.000 ₫", "tags": ["🏖️ Gần biển","🏊‍♂️ Bể bơi"]
        },
        {
            "name": "Balcony Nha Trang Hotel",
            "stars": 4, "match_score": 84.3,
            "address": "Trần Phú, Nha Trang", "reviews": 410,
            "description": "Khách sạn gần biển, phòng ban công, bữa sáng đa dạng, phù hợp nhóm bạn và gia đình...",
            "rating": 8.4, "location_score": 8.8, "cleanliness_score": 8.5, "service_score": 8.2,
            "price": "1.800.000 ₫", "tags": ["🏖️ Gần biển","🍳 Buffet sáng"]
        },
    ]

def hotels_df_for_chart(hotels):
    df = pd.DataFrame(hotels)
    # đổi tên cột để đúng nhãn trong biểu đồ
    df = df.rename(columns={
        "rating": "Điểm tổng",
        "match_score": "Độ tương đồng (%)",
        "stars": "Hạng sao"
    })
    return df
