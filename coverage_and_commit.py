import os


if __name__ == "__main__": #pragma: no cover
    test_cmd = "coverage run -m symr_tests.test_all"

    os.system(test_cmd)
    
    report_cmd = "coverage report -m > coverage.txt"

    os.system(report_cmd)

    git_add_cmd = "git add coverage.txt"

    os.system(git_add_cmd)

    with open("coverage.txt", "r") as f:

        lines = f.readlines()

    for line in lines:
        if "TOTAL" in line:
            my_commit_msg = line
            break

    git_commit_cmd = f"git commit -m 'test coverage and commit with summary: {my_commit_msg}'"

    print(git_commit_cmd)
    os.system(git_commit_cmd)


    


    
