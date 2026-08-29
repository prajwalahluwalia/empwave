import os

from empwave import create_app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5007")),
        debug=os.getenv("FLASK_DEBUG") == "1",
        use_reloader=False,
    )
