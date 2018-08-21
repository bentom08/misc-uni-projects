# B Taylor 26/01/11
# CG4 coursework program

# Import all modules needed for my program
import MySQLdb
import re
import os

# Create some useful objects; one for each table in the database

class Customer:
    def __init__(self):
        self.customerID = 0
        self.firstname = 0
        self.surname = 0
        self.address = 0
        self.email = 0
        self.telNumber = 0
        self.postCode = 0

class Calendar:
    def __init__(self):
        self.startDate = 0
        self.cost = 0
        self.isBooked = 0
        self.customerID = 0
        self.endDate = 0
        
class Payment:
    def __init__(self):
        self.paymentID = 0
        self.customerID = 0
        self.depositPaid = 0
        self.hasPaid = 0

# Connects to mySQLdb

conn = MySQLdb.connect(host = "localhost",
                       user = "ben",
                       passwd = "cat",
                       db = "ben")

# First all the menus are defined
def mainMenu():
    """
    This is the main menu from which the user can select any option and 
    go to the corresponding function
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                       |-----------------------------|                       *
*                        |Welcome to the Booking System|                      *
 *                       |-----------------------------|                       *
*                                                                             * 
 *                                                                             *
*           1. Book a Date                                                    * 
 *                                                                             *
*           2. Edit Prices in the Calendar                                    * 
 *                                                                             *
*           3. View, update or amend a booking                                * 
 *                                                                             *
*           4. Print a standard letter                                        * 
 *                                                                             *
*           5. Exit Program                                                   *
 *                                                                             *
*                                                                             * 
 *                   Please select one of the above options                    *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def bookDateMenu1():
    """
    This menu prompts the user to enter the startdate which they want to book
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                                |-----------|                                *
*                                 |Book a Date|                               *
 *                                |-----------|                                *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                  Please enter the startdate you require                     *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def bookDateMenu2():
    """
    This menu prompts the user to enter either Y or N depending on
    whether they want to book another date
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                                |-----------|                                *
*                                 |Book a Date|                               *
 *                                |-----------|                                *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                   Do you want to book another date?(Y/N)                    *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def bookDateMenu3(date):
    """
    This menu prompts the user to confirm the selected date to be booked
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                                |-----------|                                *
*                                 |Book a Date|                               *
 *                                |-----------|                                *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *            Please enter 1 to confirm booking the date""", date, """           *
*                                                                             * 
 *                      or enter 0 to cancel the booking                       *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)    
def bookDateMenu4():
    """
    This menu is a prompt to enter the customer ID or 0 depending whether the cusotmer
    is already stored in the database.
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                                |-----------|                                *
*                                 |Book a Date|                               *
 *                                |-----------|                                *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *            Please enter the customer ID if the customer details             *
*                                                                             * 
 *             are already stored in the database or 0 otherwise               *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def bookDateMenu5():
    """
    This menu is a prompt which prompts the user to enter each detail about the customer
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                                |-----------|                                *
*                                 |Book a Date|                               *
 *                                |-----------|                                *
*                                                                             * 
 *            Enter customer details in the following order:                   *
*                                                                             * 
 *                                                                             *
*         1. First Name                                                       * 
 *        2. Surname                                                           *
*         3. Address                                                          * 
 *        4. E-mail                                                            *
*         5. Telephone Number                                                 * 
 *        6. Postcode                                                          *
*                                                                             *
 *                                                                             *
*                        Press enter after each input                         * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def bookDateMenu6():
     """
     This menu is a prompt asking if the user wants to print a standard letter
     """
     print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                                |-----------|                                *
*                                 |Book a Date|                               *
 *                                |-----------|                                *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                 Do you want to print a standard letter?(Y/N)                *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def alreadyBookedMenu():
    """
    This is an error saying the date you entered has already been booked
    """
    print ("""||------------------------||
||                        ||
|| Date is already booked ||
||                        ||
||------------------------||
    """)
def invalidDateMenu():
    """
    This is an error message saying that the date entered was invalid
    """
    print ("""||--------------||
||              ||
|| Invalid Date ||
||              ||
||--------------||
    """)
def confirmBookingMenu():
    """
    This is a notification that the booking has been confirmed and entered into the database
    """
    print ("""||-------------------||
||                   ||
|| Booking Confirmed ||
||                   ||
||-------------------||
    """)
def editPricesMenu1():
    """
    This is a menu prompting to enter the startdate of the day you want to edit
    the price of
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                                |-----------|                                *
*                                 |Edit Prices|                               *
 *                                |-----------|                                *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *           Please enter the startdate of the day you want to edit            *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def editPricesMenu2():
    """
    This menu is a prompt to enter the new price for the selected date on the calendar
    """     
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                                |-----------|                                *
*                                 |Edit Prices|                               *
 *                                |-----------|                                *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                         Please enter the new price                          *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def changeBookingMenu1():
    """
    This menu is a prompt to enter which field you want to use to search
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                     |-------------------------------|                     *
*                      |View, update or amend a booking|                      *
 *                     |-------------------------------|                     *
*                                                                             *
 *            1. Customer ID                                                 * 
*             2. First Name                                                   *
 *            3. Surname                                                     * 
*             4. Address                                                      *
 *            5. Email                                                       * 
*             6. Telephone Number                                             *
 *            7. Postcode                                                    * 
*             8. If the deposit has been paid                                 *
 *            9. If the Booking has been paid for                            * 
*                                                                             *
 *                     Enter a number from 1 to 9 above to                   *
*                                                                             * 
 *                          search one of the fields                         *
*                                                                             *
 *                                                                           *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def changeBookingMenu2():
    """
    This menu is a prompt to enter the value to search
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                     |-------------------------------|                       *
*                      |View, update or amend a booking|                      *
 *                     |-------------------------------|                       *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                           Enter a value to search                           * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def changeBookingMenu3():
    """
    This subroutine is a menu that prompts the user to enter Y or N depending on
    whether or not they want to edit the booking.
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                     |-------------------------------|                       *
*                      |View, update or amend a booking|                      *
 *                     |-------------------------------|                       *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                    Do you want to edit this booking?(Y/N)                   * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def changeBookingMenu4():
    """
    This menu is a prompt to enter which detail of the booking you want to edit
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                     |-------------------------------|                     *
*                      |View, update or amend a booking|                      *
 *                     |-------------------------------|                     *
*                                                                             *
 *                                                                           * 
*             1. First Name                                                   *
 *            2. Surname                                                     * 
*             3. Address                                                      *
 *            4. Email                                                       * 
*             5. Telephone Number                                             *
 *            6. Postcode                                                    * 
*             7. If the deposit has been paid                                 *
 *            8. If the Booking has been paid for                            * 
*                                                                             *
 *                                                                           *
*                Enter the number of the field you want to edit               * 
 *                                                                           *
*                                                                             *
 *                                                                           *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def changeBookingMenu5():
    """
    This menu is a prompt to enter the new value for a booking
    """
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                     |-------------------------------|                       *
*                      |View, update or amend a booking|                      *
 *                     |-------------------------------|                       *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                             Enter the new value                             * 
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
*                                                                             * 
 *                                                                             *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
def changeBookingMenu6(myCustomer, myPayment, myDate):
    """
    This subroutine prints a menu with all the cusotmer details and all the details
    of the booking.
    """
    if myPayment.depositPaid == 0: # change the depositPaid variable into Yes or No rather than 1 or 0 
        myPayment.depositPaid = "No"
    elif myPayment.depositPaid == 1:
        myPayment.depositPaid = "Yes"
    if myPayment.hasPaid == 0: # changes hasPaid to Yes or No
        myPayment.hasPaid = "No"
    elif myPayment.hasPaid == 1:
        myPayment.hasPaid = "Yes"
    print ("""* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                             *
 *                     |-------------------------------|                       *
*                      |View, update or amend a booking|                      *
 *                     |-------------------------------|                       *
*                                                                             * 
 *            Start Date      :""", myDate.startDate, """ 
*             Customer ID     :""", myCustomer.customerID, """
 *            First Name      :""", myCustomer.firstname, """
*             Surname         :""", myCustomer.surname, """ 
 *            Address         :""", myCustomer.address, """
*             E-mail          :""", myCustomer.email, """ 
 *            Telephone Number:""", myCustomer.telNumber, """
*             Postcode        :""", myCustomer.postCode, """ 
 *            Deposit Paid?   :""", myPayment.depositPaid, """
*             Booking Paid?   :""", myPayment.hasPaid, """
 *                                                                             *
*                                                                             * 
 *                           Press Enter to continue                           *
*                                                                             *
 *                                                                             *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     """)
     
# Now the main program starts
def getDateDetails(menu):
    """
    This function searches for a date in the database and gets all the details
    from that date. Arguments - menu which determines which menu to print
    as a prompt. It return an array - line - containing all the variables from
    the date the user entered.
    """
    myDate = Calendar() # create myDate object
    
    while myDate.startDate == 0:
        # display the menu for booking, request a date and get our results
        if menu == 1:
            bookDateMenu1() # menu prompting the user to enter a date
        elif menu == 2:
            editPricesMenu1() # menu prompting the user to enter a date
        startDate = int(raw_input("(Enter in the format YYYYMMDD)")) # date required is entered and searched for
        sql = "SELECT * FROM Calendar WHERE Startdate = %d" % (startDate)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        # store data from database in temporary list
        for line in rows:
            myDate.startDate = line[0]
            myDate.cost = line[1]
            myDate.isBooked = line[2]
            myDate.bookingID = line[3]
            myDate.endDate = line[4]
        # we haven't got a valid date print an error message
        if myDate.startDate == 0:
            invalidDateMenu()

    # return date details in an object
    return myDate
def validateEmail(email):
    """
    This function validates the email address entered. It accepts the argument for the
    email address and returns either 1 or 0 depending on whether the address is valid.
    """
    if len(email) > 6 and len(email) <= 50:
        if re.match("^[a-zA-Z0-9._%\-+]+@[a-zA-Z0-9._%-]+.[a-zA-Z]{2,6}$", email) != None: # regular expression to validate
            return 1
    return 0
def validateTelNumber(telNumber):
    """
    This validates the telephone number. Its argument is the telephone number that needs
    to be validated and it returns 1 or 0 depending whether the number is valid.
    """
    if len(telNumber) <= 20:
        if re.match("^[(0]{1,2}[)0-9]+[- 0-9]{5,11}$", telNumber) != None:
            return 1
    return 0
def validatePostcode(postcode):
    """
    This validates the post code. The argument is the post code that needs to be validated
    and it returns 1 or 0 depending on whether it is valid or not
    """
    if len(postcode) <= 8:
        if re.match("^[a-zA-Z0-9 ]{5,8}$", postcode) != None:
            return 1
    return 0
def validateDetails(myCustomer):
    """
    This function validates all inputs about the customer by checking if the first name,
    surname and address are below the mySQLdb field length and checking the email,
    telephone number and post code are valid by using the regular expressions in the
    functions above. The argument is the object myCustomer, which includes all the details.
    it returns 1 if all details are valid, or 0 if one or more are invalid.
    """
    if len(myCustomer.firstname) >= 25:
        return 2
    if len(myCustomer.surname) >= 25:
        return 3
    if len(myCustomer.address) >= 100:
        return 4
    if validateEmail(myCustomer.email) == 0:
        return 5
    if validateTelNumber(myCustomer.telNumber) == 0:
        return 6
    if validatePostcode(myCustomer.postCode) == 0:
        return 7
    return 1
def getCustomerDetails():
    """
    This function asks the user for a customer ID or the details of a new customer to be
    entered into the database. It has no arguments and returns the object myCustomer which
    contains all the details of the customer.
    """
    bookDateMenu4() # prompts the user to enter a customer ID or 0 depending on whether the customer is already in the database
    while True: # This validates the customer ID entered is an integer containing less than 4 digits
        try:
            customerID = int(raw_input())
            if len(str(customerID)) < 5:
                break
            print ("Please enter a number between 0 and 9999")
        except:
            print ("Please enter a number between 0 and 9999")
    myCustomer = Customer() # Creates the myCustomer object
    if customerID == 0:
        detailsValidated = False
        while detailsValidated == False:
            bookDateMenu5() # prompts the user to enter all customer details
            # get all customer details in the myCustomer object
            myCustomer.firstname = raw_input("First Name --> ")
            myCustomer.surname = raw_input("Surname --> ")
            myCustomer.address = raw_input("Address --> ")
            myCustomer.email = raw_input("Email --> ")
            myCustomer.telNumber = raw_input("Telephone Number --> ")
            myCustomer.postCode = raw_input("Post Code --> ")
            if validateDetails(myCustomer) == 1: # checks all the details are valid
                detailsValidated = True
            # if not an error message is printed telling the user which input was invalid
            elif validateDetails(myCustomer) == 2:
                print ("The First Name was too long, please enter everything again")
            elif validateDetails(myCustomer) == 3:
                print ("The Surname was too long, please enter everything again")
            elif validateDetails(myCustomer) == 4:
                print ("The Address was too long, please enter everthing again")
            elif validateDetails(myCustomer) == 5:
                print ("The Email address is invalid, please enter everything again")
            elif validateDetails(myCustomer) == 6:
                print ("The Telephone Number is invalid, please enter everything again")
            elif validateDetails(myCustomer) == 7:
                print ("The post code is invalid, please enter everything again")
        sql = "INSERT INTO Customer (firstName, surname, ADDRESS, Email, TelNumber, postCode) VALUES ('%s', '%s', '%s', '%s', '%s', '%s')" % (myCustomer.firstname, myCustomer.surname, myCustomer.address, myCustomer.email, myCustomer.telNumber, myCustomer.postCode)
        cursor = conn.cursor()
        cursor.execute(sql) # enters all details inputed into the mySQLb
        sql = "SELECT * FROM Customer WHERE TelNumber = '%s'" % (myCustomer.telNumber)
        cursor = conn.cursor()
        cursor.execute(sql) 
        rows = cursor.fetchall()
        #assigns all the customer details to myCustomer
        for line in rows:
            myCustomer.customerID = line[0]
        return myCustomer # returns the details as an object
    else:
        sql = "SELECT * FROM Customer WHERE CustomerID = %i" % (customerID)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        # retrieves all details from the selected customer ID and assigns them to the myCustomer object
        for line in rows:
            myCustomer.customerID = line[0]
            myCustomer.firstname = line[1]
            myCustomer.surname = line[2]
            myCustomer.address = line[3]
            myCustomer.email = line[4]
            myCustomer.telNumber = line[5]
            myCustomer.postCode = line[6]
        return myCustomer
def bookDate():
    """    
    Subroutine that handles the date book process
    """
    bookingIncomplete = True
    
    while bookingIncomplete:
        # get our date details
        myDate = getDateDetails(1)
        flag = False
        #checks if the date searched for is already booked
        if myDate.isBooked == 1:
            alreadyBookedMenu() #error message if the date is already booked
            searchAgain = raw_input("Do you want to search for another date?(Y/N) ")
            if searchAgain == "Y" or searchAgain == "y":
                flag = True
            else:
                break
           
        if flag == False:
            #this converts the date into an easy to read format
            startDate = str(myDate.startDate)
            startDate = startDate[8:10] + startDate[7] + startDate[5:7] + startDate[4] + startDate[0:4]
            bookDateMenu3(startDate) # Asks the user to press 1 to confirm booking the date or 0 to cancel
            confirmDate = raw_input()
            #thsi converts the date into a format that can be searched for by the mySQLdb
            startDate = str(myDate.startDate)
            startDate = int(startDate[0:4] + startDate[5:7] + startDate[8:10])
            if confirmDate == "1":
                sql = "UPDATE Calendar SET isBooked = 1 WHERE Startdate = %d" % (startDate)
                cursor = conn.cursor()
                cursor.execute(sql) #updates the database so the date is booked
                confirmBookingMenu() # menu confirming the date has been booked
            else:
                break
            #gets the customer details and stores them in the myCustomer object
            myCustomer = getCustomerDetails()
            #adds the customer ID to payment database and Calendar database
            sql = "UPDATE Calendar SET CustomerID = %i WHERE Startdate = %d" % (myCustomer.customerID, startDate)
            cursor = conn.cursor()
            cursor.execute(sql)
            sql = "INSERT INTO Payment (customerID) VALUES (%i)" % (myCustomer.customerID)
            cursor = conn.cursor()
            cursor.execute(sql)
            bookDateMenu6() # Asks the user to enter Y to print a standard letter or N not to
            letterChoice = raw_input()
            if letterChoice == "Y" or letterChoice == "y":
                printStandardLetter(myCustomer, myDate) # prints the standard letter
            bookDateMenu2() # menu saying enter Y to book another date or N to go back to main menu
            bookAgain = raw_input()
            if bookAgain == "Y" or bookAgain == "y":
                pass
            else:
                break
def printStandardLetter(myCustomer, myDate):
    """
    This subroutine prints a standard letter by replacing certains tags in a text file with
    variables, writing it to a different text file and printing it off. Arguments are
    the myCustomer and myDate objects.
    """
    # converts the startdate and endDate into formats that are easy to read
    myDate.startDate = str(myDate.startDate)
    myDate.startDate = myDate.startDate[8:10] + myDate.startDate[7] + myDate.startDate[5:7] + myDate.startDate[4] + myDate.startDate[0:4]
    myDate.endDate = str(myDate.endDate)
    myDate.endDate = myDate.endDate[8:10] + myDate.endDate[7] + myDate.endDate[5:7] + myDate.endDate[4] + myDate.endDate[0:4]
    # opens the file with the letter and assigns it to the letter variable
    f = open('standardLetter.txt', 'r')
    letter = f.read()
    f.close()
    # replaces the tags with customer details
    letter = letter.replace("<fname>", myCustomer.firstname)
    letter = letter.replace("<startDate>", myDate.startDate)
    letter = letter.replace("<endDate>", myDate.endDate)
    letter = letter.replace("<cost>", str(myDate.cost))
    #writes the details to a new text file
    f = open('modLetter.txt', 'w')
    f.write(letter)
    f.close()
    #prints the new letter
    os.system('lpr modLetter.txt')

def editPrices():
    """
    This is a subroutine that handles editing prices in the database
    """
    myDate = Calendar() # create the myDate object
    myDate = getDateDetails(2) # assign date details to myDate object
    print ("Current Price is \xa3" + str(myDate.cost))
    editPricesMenu2() # prompt for the user to enter the new price
    # validates that the newCost is an integer
    while True:
        newCost = raw_input()
        if re.match("^[0-9]{1,5}$", newCost) != None:
            break
        else:
            print ("Invalid Price, please enter again")
    # converts the date into a format that can be entered into the database
    startDate = str(myDate.startDate)
    startDate = int(startDate[0:4] + startDate[5:7] + startDate[8:10])
    sql = "UPDATE Calendar SET Cost = '%s' WHERE Startdate = %d" % (newCost, startDate)
    cursor = conn.cursor()
    cursor.execute(sql) # updates the database to the new cost

def search(myCustomer, myDate, myPayment):
    """
    This function takes the customer, date and payment objects as arguments. It
    searches for the entered details and returns the results in the myCustomer,
    myDate and myPayment objects.
    """
    changeBookingMenu1() # Menu prompting the user to enter 1-9 depending on which field they want to search
    # Validates that the searchOption is an integer
    while True:
        try:
            searchOption = int(raw_input())
            break
        except:
            print ("Please enter a valid input")
    changeBookingMenu2() # Menu prompting user to enter the value they want to search
    # Validates that if option 8 or 9 are searched for the input is either Y or N
    if searchOption == 8 or searchOption == 9:
        while True:
            value = raw_input()
            if value == "y" or value == "Y" or value == "n" or value == "N":
                break
            else:
                print ("Enter either Y or N")
    else:
        value = raw_input()
    # selects all details from the customer database for the searched customer
    if searchOption == 1:
        sql = "SELECT * FROM Customer WHERE CustomerID = '%s'" % (value)
    elif searchOption == 2:
        sql = "SELECT * FROM Customer WHERE firstname LIKE '%s'" % (value)
    elif searchOption == 3:
        sql = "SELECT * FROM Customer WHERE surname LIKE '%s'" % (value)
    elif searchOption == 4:
        sql = "SELECT * FROM Customer WHERE ADDRESS LIKE '%s'" % (value)
    elif searchOption == 5:
        sql = "SELECT * FROM Customer WHERE Email LIKE '%s'" % (value)
    elif searchOption == 6:
        sql = "SELECT * FROM Customer WHERE TelNumber LIKE '%s'" % (value)
    elif searchOption == 7:
        sql = "SELECT * FROM Customer WHERE postCode LIKE '%s'" % (value)
    elif searchOption == 8 or searchOption == 9:
        # changes the y or n to a 1 or 0 so it can be entered into the database
        if value == "y" or value == "Y":
            value = 1
        elif value == "n" or value == "N":
            value = 0
        if searchOption == 8:
            sql = "SELECT * FROM Payment WHERE depositPaid = %i" % (value)
        elif searchOtion == 9:
            sql = "SELECT * FROM Payment WHERE hasPaid = %i" % (value)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        # assigns the depositPaid and hasPaid to their variables
        for line in rows:
            myPayment.customerID = line[1]
            myPayment.depositPaid = line[2]
            myPayment.hasPaid = line[3]
        sql = "SELECT * FROM Customer WHERE customerID = %i" % (myPayment.customerID)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    # assigns all customer details to variables
    for line in rows:
        myCustomer.customerID = line[0]
        myCustomer.firstname = line[1]
        myCustomer.surname = line[2]
        myCustomer.address = line[3]
        myCustomer.email = line[4]
        myCustomer.telNumber = line[5]
        myCustomer.postCode = line[6]
    # If the payment details weren't already assigned previously, they are assigned here
    if searchOption != 8 and searchOption != 9:
        sql = "SELECT * FROM Payment WHERE customerID = %i" % (myCustomer.customerID)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        for line in rows:
            myPayment.depositPaid = line[2]
            myPayment.hasPaid = line[3]
    sql = "SELECT * FROM Calendar WHERE customerID = %i" % (myCustomer.customerID)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    # assigns the startDate to a variable
    for line in rows:
        myDate.startDate = line[0]
        myDate.cost = line[1]
        myDate.endDate = line[4]
    # prints a menu with all the details
    changeBookingMenu6(myCustomer, myPayment, myDate) # Menu which prints details of the customer that was searched for
    raw_input()
    return myCustomer, myPayment, myDate

def changeBooking():
    """
    This subroutine handles changing any details of a booking. It has arguments
    for the myCustomer, myPayment and myDate objects.
    """
    myCustomer = Customer()
    myPayment = Payment()
    myDate = Calendar()
    Customer, myPayment, myDate = search(myCusomter, myDate, myPayment)
    changeBookingMenu3() # asks the user if they want to edit the selected booking
    # validates that editChoice is either y or n
    while True:
        editChoice = raw_input()
        if editChoice == "Y" or editChoice == "N" or editChoice == "y" or editChoice == "n":
            break
        else:
            print ("Please enter either Y or N")
    if editChoice == "Y" or editChoice == "y":
        changeBookingMenu4() # Menu prompting user to enter a number from 1-8 to edit the corresponding field
        editField = raw_input()
        # if email is changed it makes sure the input is a valid email
        if editField == "4":
            while True:
                changeBookingMenu5() # Menu prompting the user to enter the new value for the booking
                newValue = raw_input()
                if validateEmail(newValue) == 1:
                    break
                else:
                    print ("Please Enter a Valid E-mail")
        # if hasPaid or depositPaid is changed it makes sure the input is either Y or N
        elif editField == "7" or editField == "8":
            while True:
                changeBookingMenu5() #  Menu prompting the user to enter the new value
                newValue = raw_input()
                if newValue == "Y" or newValue == "y" or newValue == "N" or newValue == "n":
                    break
                else:
                    print ("Please enter either Y or N")
            # changes the Y or N to 1 or 0 so they can be entered into the database
            if newValue == "Y" or newValue == "y":
                newValue = 1
            elif newValue == "N" or newValue == "n":
                newValue = 0
        else:
            changeBookingMenu5() # Menu prompting the user to enter the new value
            newValue = raw_input()
        # updates the database with the new value
        if editField == "1":
            sql = "UPDATE Customer SET firstName = '%s' WHERE CustomerID = %i" % (newValue, myCustomer.customerID)
        elif editField == "2":
            sql = "UPDATE Customer SET surname = '%s' WHERE CustomerID = %i" % (newValue, myCustomer.customerID)
        elif editField == "3":
            sql = "UPDATE Customer SET ADDRESS = '%s' WHERE CustomerID = %i" % (newValue, myCustomer.customerID)
        elif editField == "4":
            sql = "UPDATE Customer SET Email = '%s' WHERE CustomerID = %i" % (newValue, myCustomer.customerID)
        elif editField == "5":
            sql = "UPDATE Customer SET TelNumber = '%s' WHERE CusotmerID = %i" % (newValue, myCustomer.customerID)
        elif editField == "6":
            sql = "UPDATE Customer SET postCode = '%s' WHERE CustomerID = %i" % (newValue, myCustomer.customerID)
        elif editField == "7":
            sql = "UPDATE Payment SET depositPaid = %i WHERE customerID = %i" % (newValue, myCustomer.customerID)
        elif editField == "8":
            sql = "UPDATE Payment SET hasPaid = %i WHERE customerID = %i" % (newValue, myCustomer.customerID)
        cursor = conn.cursor()
        cursor.execute(sql)
        sql = "SELECT * FROM Customer WHERE CustomerID = %i" % (myCustomer.customerID)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        # assigns all updated customer details to variables
        for line in rows:
            myCustomer.customerID = line[0]
            myCustomer.firstname = line[1]
            myCustomer.surname = line[2]
            myCustomer.address = line[3]
            myCustomer.email = line[4]
            myCustomer.telNumber = line[5]
            myCustomer.postCode = line[6]
        sql = "SELECT * FROM Payment WHERE customerID = %i" % (myCustomer.customerID)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        # assigns all payment details to variables
        for line in rows:
            myPayment.depositPaid = line[2]
            myPayment.hasPaid = line[3]
        sql = "SELECT Startdate FROM Calendar WHERE customerID = %i" % (myCustomer.customerID)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        # assigns the startdate to a variable
        for line in rows:
            myDate.startDate = line[0]
        # prints the menu with the updated details on it
        changeBookingMenu6(myCustomer, myPayment, myDate) # Menu with all the updated details from the booking
        raw_input()    

#main
while True:
    mainMenu()
    option = raw_input()
    if option == "1":
        # book a new date
        bookDate()
    elif option == "2":
        # edit a price in the calendar
        editPrices()
    elif option == "3":
        # change a booking
        changeBooking()
    elif option == "4":
        # prints a standard letter
        myCustomer = Customer()
        myPayment = Payment()
        myDate = Calendar()
        myCustomer, myPayment, myDate = search(myCustomer, myDate, myPayment)
        bookDateMenu6() # menu asking the user if they want to print a standard letter
        while True:
            printLetter = raw_input()
            if printLetter == "Y" or printLetter == "N" or printLetter == "y" or printLetter == "n":
                break
            else:
                print ("Please enter either Y or N")
        if printLetter == "Y" or printLetter == "y":
            printStandardLetter(myCustomer, myDate)
        else:
            pass
    elif option == "5":
        exit()
    else:
        print ("Invalid choice, please enter a valid option")