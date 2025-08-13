from setup import run_setup

if __name__ == "__main__":
    run_setup(
        package_name="hypertools",
        requires=[],
        package_data={"": ["py.typed"], "loafpan": ["logging/static/*.*"]},
    )
