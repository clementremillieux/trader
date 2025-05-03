from coinbase import jwt_generator

secret = {
    "name": "bb54e080-d370-4bd0-b0cd-4f3c651951c9",
    "privateKey": "/HddxeryxZaqzCT4U0w1IzXvqyx5C647pyTD7eABWHhhXXuK/vJhz9oBjGfuFn4GCTv1jlZFvmOe+/GiQlDsOg==",
}

api_key = f"organizations/{secret['name']}/apiKeys/{secret['privateKey']}"

api_secret = (
    f"-----BEGIN EC PRIVATE KEY-----\n{api_key}\n-----END EC PRIVATE KEY-----\n"
)

request_method = "GET"
request_path = "/v2/accounts"


def main():
    jwt_uri = jwt_generator.format_jwt_uri(request_method, request_path)
    jwt_token = jwt_generator.build_rest_jwt(jwt_uri, api_key, api_secret)
    print(f"export JWT={jwt_token}")


if __name__ == "__main__":
    main()
