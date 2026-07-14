#include <gtest/gtest.h>

#include "reference.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    mfxtest::print_report();
    return rc;
}
