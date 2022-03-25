
import requests
from prod_reco.commons.recommender_utils import USER_ID, ITEM_ID


class TestEndpoint:
    endpoint = "http://127.0.0.1:5001/invocations"
    # Item ID
    # 1558 = Dark Knight
    # 1440 = Kung Fu Panda
    # 418 = Godfather II
    # User ID
    # 2241 = Classics, a little bit of crime
    # 190 = Very into Animation & Fantasy
    def test_single_item(self):
        payload = {
            "columns": [ITEM_ID],
            "data": [[1558]]
        }
        response = requests.post(self.endpoint, json=payload)
        print()
        print("Dark Knight")
        print(response.text)
        assert response.status_code == 200

    def test_two_items(self):
        payload = {
            "columns": [ITEM_ID],
            "data": [[1558], [1440]]
        }

        response = requests.post(self.endpoint, json=payload)
        print()
        print("Dark Knight & Kung Fu Panda")
        print(response.text)
        assert response.status_code == 200

    def test_three_items(self):
        payload = {
            "columns": [ITEM_ID],
            "data": [[1558], [1440], [418]]
        }

        response = requests.post(self.endpoint, json=payload)
        print()
        print("Dark Knight & Kung Fu Panda & Godfather II")
        print(response.text)
        assert response.status_code == 200

    def test_two_items_and_user(self):
        payload = {
            "columns": [USER_ID, ITEM_ID],
            "data": [[2241, 1558], [2241, 1440]]
        }

        response = requests.post(self.endpoint, json=payload)
        print()
        print("User likes vintage movies")
        print(response.text)
        assert response.status_code == 200

    
    def test_two_items_per_user(self):
        payload = {
            "columns": [USER_ID, ITEM_ID],
            "data": [[2241, 1558], [2241, 1440], [190, 1558], [190, 1440]]
        }

        response = requests.post(self.endpoint, json=payload)
        print()
        print("First user likes vintage movies, second likes animation")
        print(response.text)
        assert response.status_code == 200