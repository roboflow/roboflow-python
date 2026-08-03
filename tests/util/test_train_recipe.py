import unittest

from roboflow.util.train_recipe import fold_epochs_into_recipe


class TestFoldEpochsIntoRecipe(unittest.TestCase):
    def test_epochs_folded_into_hyperparameters(self):
        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.0002}}
        folded = fold_epochs_into_recipe(recipe, 50)
        self.assertEqual(folded["hyperparameters"], {"lr": 0.0002, "epochs": 50})
        self.assertEqual(folded["schema_version"], 1)

    def test_epochs_does_not_clobber_explicit_hyperparameter(self):
        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.0002, "epochs": 25}}
        folded = fold_epochs_into_recipe(recipe, 50)
        self.assertEqual(folded["hyperparameters"]["epochs"], 25)

    def test_epochs_fold_creates_missing_hyperparameters_key(self):
        # Hand-written recipes may omit the hyperparameters key entirely.
        folded = fold_epochs_into_recipe({"schema_version": 1}, 50)
        self.assertEqual(folded["hyperparameters"], {"epochs": 50})

    def test_input_recipe_is_not_mutated(self):
        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.0002}}
        fold_epochs_into_recipe(recipe, 50)
        self.assertEqual(recipe, {"schema_version": 1, "hyperparameters": {"lr": 0.0002}})


if __name__ == "__main__":
    unittest.main()
