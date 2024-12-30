from data import students  # data.py에서 students 데이터를 임포트

def average_score():
    total_score = sum(students.values())  # students의 점수 합
    num_students = len(students)  # 학생 수
    return total_score / num_students  # 평균 점수 계산

def best_student():
    best_score = max(students.values())  # 최고 점수
    best_student_name = [name for name, score in students.items() if score == best_score]  # 최고 점수를 받은 학생 이름
    return best_student_name[0]  # 가장 높은 점수의 첫 번째 학생 이름 반환

# 사용 예시
if __name__ == "__main__":
    print(f"평균 점수: {average_score()}")  # 평균 점수 출력
    print(f"최고 점수 학생: {best_student()}")  # 최고 점수를 받은 학생 이름 출력