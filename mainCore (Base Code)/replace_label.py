import csv

source_csv = 'model/keypoint_classifier/keypoint.csv'


def copy_and_update_labels(label):
    target_csv = 'dynamic.csv'
    rows_to_copy = []

    with open(source_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == str(label):
                rows_to_copy.append(row)

    with open(target_csv, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_to_copy)

    return target_csv


def change_label_in_csv(csv_path, old_label, new_label):
    updated_rows = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == str(old_label):
                row[0] = str(new_label)
            updated_rows.append(row)

    with open(csv_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)


def replace_number_in_csv(old_number, new_number):
    updated_rows = []

    with open(source_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == str(old_number):
                row[0] = str(new_number)
            updated_rows.append(row)

    with open(source_csv, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)


def delete_number_in_csv(number):
    updated_rows = []

    with open(source_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] != str(number):
                updated_rows.append(row)

    with open(source_csv, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)


def menu():
    print("\n--- Menu ---")
    print("1. Copy and update labels")
    print("2. Replace a specific number with a new number")
    print("3. Delete a specific number")
    print("4. Exit")
    choice = input("\nEnter your choice (1/2/3/4): ").strip()
    return choice


def main():
    while True:
        choice = menu()
        if choice == '1':
            label = int(input("\nEnter the label (number) you want to copy: "))
            target_csv = copy_and_update_labels(label)
            print(f"\nCopied all rows with label {label} to {target_csv}")

            change_label = input("\nDo you want to change the label in the new CSV? (y/n): ").strip().lower()
            if change_label == 'y':
                new_label = int(input("Enter the new label: "))
                change_label_in_csv(target_csv, label, new_label)
                print(f"\nChanged label {label} to {new_label} in {target_csv}")

        elif choice == '2':
            old_number = int(input("\nEnter the number you want to replace: "))
            new_number = int(input("Enter the new number: "))
            replace_number_in_csv(old_number, new_number)
            print(f"\nReplaced all occurrences of {old_number} with {new_number} in {source_csv}")

        elif choice == '3':
            number_to_delete = int(input("\nEnter the number you want to delete: "))
            confirm = input(f"Are you sure you want to delete all occurrences of {number_to_delete}? (y/n): ").strip().lower()
            if confirm == 'y':
                delete_number_in_csv(number_to_delete)
                print(f"\nDeleted all occurrences of {number_to_delete} from {source_csv}")
            else:
                print("\nDeletion cancelled. Returning to menu.")

        elif choice == '4':
            print("\nExiting the program. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
