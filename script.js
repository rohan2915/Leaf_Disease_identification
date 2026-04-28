// Task Manager Application
class TaskManager {
    constructor() {
        this.tasks = JSON.parse(localStorage.getItem('tasks')) || [];
        this.currentFilter = 'all';
        this.editingTaskId = null;
        
        this.initializeElements();
        this.bindEvents();
        this.renderTasks();
        this.updateStats();
    }

    initializeElements() {
        // Forms
        this.taskForm = document.getElementById('taskForm');
        this.editForm = document.getElementById('editForm');
        
        // Inputs
        this.taskTitleInput = document.getElementById('taskTitle');
        this.taskDescriptionInput = document.getElementById('taskDescription');
        this.editTitleInput = document.getElementById('editTitle');
        this.editDescriptionInput = document.getElementById('editDescription');
        
        // Containers
        this.taskList = document.getElementById('taskList');
        this.emptyState = document.getElementById('emptyState');
        
        // Stats
        this.totalTasksElement = document.getElementById('totalTasks');
        this.activeTasksElement = document.getElementById('activeTasks');
        this.completedTasksElement = document.getElementById('completedTasks');
        
        // Modal
        this.editModal = document.getElementById('editModal');
        this.closeModalBtn = document.getElementById('closeModal');
        this.cancelEditBtn = document.getElementById('cancelEdit');
        
        // Filters
        this.filterButtons = document.querySelectorAll('.filter-btn');
    }

    bindEvents() {
        // Task form submission
        this.taskForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.addTask();
        });

        // Edit form submission
        this.editForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveEdit();
        });

        // Modal events
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        this.cancelEditBtn.addEventListener('click', () => this.closeModal());
        
        // Close modal when clicking outside
        this.editModal.addEventListener('click', (e) => {
            if (e.target === this.editModal) {
                this.closeModal();
            }
        });

        // Filter buttons
        this.filterButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setFilter(e.target.dataset.filter);
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }

    addTask() {
        const title = this.taskTitleInput.value.trim();
        const description = this.taskDescriptionInput.value.trim();
        
        if (!title) return;

        const task = {
            id: Date.now().toString(),
            title,
            description,
            completed: false,
            createdAt: new Date().toISOString()
        };

        this.tasks.unshift(task);
        this.saveToLocalStorage();
        this.renderTasks();
        this.updateStats();
        this.clearForm();
        
        // Success animation
        this.showSuccessMessage('Task added successfully!');
    }

    editTask(taskId) {
        const task = this.tasks.find(t => t.id === taskId);
        if (!task) return;

        this.editingTaskId = taskId;
        this.editTitleInput.value = task.title;
        this.editDescriptionInput.value = task.description;
        this.openModal();
    }

    saveEdit() {
        const title = this.editTitleInput.value.trim();
        const description = this.editDescriptionInput.value.trim();
        
        if (!title) return;

        const taskIndex = this.tasks.findIndex(t => t.id === this.editingTaskId);
        if (taskIndex === -1) return;

        this.tasks[taskIndex].title = title;
        this.tasks[taskIndex].description = description;
        this.tasks[taskIndex].updatedAt = new Date().toISOString();

        this.saveToLocalStorage();
        this.renderTasks();
        this.closeModal();
        
        this.showSuccessMessage('Task updated successfully!');
    }

    deleteTask(taskId) {
        if (confirm('Are you sure you want to delete this task?')) {
            this.tasks = this.tasks.filter(t => t.id !== taskId);
            this.saveToLocalStorage();
            this.renderTasks();
            this.updateStats();
            
            this.showSuccessMessage('Task deleted successfully!');
        }
    }

    toggleTaskComplete(taskId) {
        const task = this.tasks.find(t => t.id === taskId);
        if (!task) return;

        task.completed = !task.completed;
        task.completedAt = task.completed ? new Date().toISOString() : null;
        
        this.saveToLocalStorage();
        this.renderTasks();
        this.updateStats();
        
        const message = task.completed ? 'Task marked as complete!' : 'Task marked as active!';
        this.showSuccessMessage(message);
    }

    setFilter(filter) {
        this.currentFilter = filter;
        
        // Update active filter button
        this.filterButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });
        
        this.renderTasks();
    }

    renderTasks() {
        const filteredTasks = this.getFilteredTasks();
        
        if (filteredTasks.length === 0) {
            this.taskList.style.display = 'none';
            this.emptyState.style.display = 'block';
        } else {
            this.taskList.style.display = 'block';
            this.emptyState.style.display = 'none';
            
            this.taskList.innerHTML = filteredTasks.map(task => this.createTaskHTML(task)).join('');
            
            // Bind events to newly created elements
            this.bindTaskEvents();
        }
    }

    getFilteredTasks() {
        switch (this.currentFilter) {
            case 'active':
                return this.tasks.filter(task => !task.completed);
            case 'completed':
                return this.tasks.filter(task => task.completed);
            default:
                return this.tasks;
        }
    }

    createTaskHTML(task) {
        const completedClass = task.completed ? 'completed' : '';
        const checkedClass = task.completed ? 'checked' : '';
        
        return `
            <div class="task-item ${completedClass}" data-task-id="${task.id}">
                <div class="task-header">
                    <div class="task-checkbox ${checkedClass}" data-task-id="${task.id}"></div>
                    <div class="task-title">${this.escapeHtml(task.title)}</div>
                    <div class="task-actions">
                        <button class="action-btn edit-btn" data-task-id="${task.id}">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="action-btn delete-btn" data-task-id="${task.id}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                ${task.description ? `<div class="task-description">${this.escapeHtml(task.description)}</div>` : ''}
            </div>
        `;
    }

    bindTaskEvents() {
        // Checkbox events
        document.querySelectorAll('.task-checkbox').forEach(checkbox => {
            checkbox.addEventListener('click', (e) => {
                const taskId = e.target.dataset.taskId;
                this.toggleTaskComplete(taskId);
            });
        });

        // Edit button events
        document.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const taskId = e.target.closest('.edit-btn').dataset.taskId;
                this.editTask(taskId);
            });
        });

        // Delete button events
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const taskId = e.target.closest('.delete-btn').dataset.taskId;
                this.deleteTask(taskId);
            });
        });
    }

    updateStats() {
        const total = this.tasks.length;
        const completed = this.tasks.filter(t => t.completed).length;
        const active = total - completed;

        this.totalTasksElement.textContent = total;
        this.activeTasksElement.textContent = active;
        this.completedTasksElement.textContent = completed;
    }

    clearForm() {
        this.taskTitleInput.value = '';
        this.taskDescriptionInput.value = '';
        this.taskTitleInput.focus();
    }

    openModal() {
        this.editModal.classList.add('show');
        this.editTitleInput.focus();
    }

    closeModal() {
        this.editModal.classList.remove('show');
        this.editingTaskId = null;
        this.editTitleInput.value = '';
        this.editDescriptionInput.value = '';
    }

    saveToLocalStorage() {
        localStorage.setItem('tasks', JSON.stringify(this.tasks));
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showSuccessMessage(message) {
        // Create temporary success message
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.textContent = message;
        successDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1001;
            animation: slideIn 0.3s ease;
        `;

        document.body.appendChild(successDiv);

        // Remove after 3 seconds
        setTimeout(() => {
            successDiv.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (successDiv.parentNode) {
                    successDiv.parentNode.removeChild(successDiv);
                }
            }, 300);
        }, 3000);
    }
}

// Add CSS animations for success messages
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TaskManager();
});

// Add some sample tasks for demonstration (remove in production)
if (!localStorage.getItem('tasks')) {
    const sampleTasks = [
        {
            id: '1',
            title: 'Welcome to AI Task Manager!',
            description: 'This is your first task. Click the checkbox to mark it as complete.',
            completed: false,
            createdAt: new Date().toISOString()
        },
        {
            id: '2',
            title: 'Explore the features',
            description: 'Try adding, editing, and deleting tasks. Use the filters to organize your view.',
            completed: false,
            createdAt: new Date().toISOString()
        }
    ];
    localStorage.setItem('tasks', JSON.stringify(sampleTasks));
} 